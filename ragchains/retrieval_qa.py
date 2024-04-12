import hashlib
from typing import List
import uuid
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnablePick, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from unstructured.documents.elements import Text, Table

from langchain_core.documents import Document


from .ingest import ingest_document
from .store import RetrievalStore


CACHE_DIR = "documents/"
needs_text_summary = False
needs_table_summary = False


def print_me(inputs):
    print("inputs:", inputs)
    return inputs


def summarize_text_block(text_element: Text, chat_model: BaseChatModel, text_id: str):
    text = text_element.text
    # prompt = ChatPromptTemplate.from_template("The following content is an excerpt from an insurance report. Give a concise summary of the following content, retain important information. Also state the report title, if present: {data}")
    # summarize_chain = {"data": RunnablePassthrough()} | prompt | chat_model | StrOutputParser()
    # print("Summarizing text %s, \"%s\"..." % (text_id, text[:30]))
    
    # if "November 2, 2022" in text:
    #     text = text.replace("November 2, 2022", "<Report Date> November 2, 2022")
    
    return text  # summarize_chain.invoke(text)


def summarize_table(table_html: str, chat_model: BaseChatModel, table_id: str):
    prompt = ChatPromptTemplate.from_template("The following content is an excerpt from an insurance report. Give a concise summary of the following HTML table in English, retain important information: ```{data}```")
    summarize_chain = {"data": RunnablePassthrough()} | prompt | chat_model | StrOutputParser()
    print("Summarizing table %s..." % table_id)
    return summarize_chain.invoke(table_html)


def ingest_process(document_file, store: RetrievalStore, summarizer_model: BaseChatModel = None, cache_directory: str = None):
    print("Ingesting documents...")
    parsed_doc = ingest_document(document_file, cache_directory=cache_directory)

    retriever = store.retriever
    vectorstore = store.vectorstore

    if isinstance(retriever, MultiVectorRetriever):
        id_key = store.id_key

        # Text
        text_elements = list(filter(lambda x: isinstance(x, Text), parsed_doc))
        texts = list(map(str, text_elements))
        text_ids = [str(uuid.UUID(bytes=hashlib.sha256(t.encode()).digest()[:16])) for t in texts]

        # Store original text copy
        retriever.docstore.mset(list(zip(text_ids, texts)))

        if needs_text_summary:
            print("Saving text summaries...")
            summary_texts = [
                Document(page_content=summarize_text_block(s, summarizer_model, t_id), metadata={id_key: t_id})
                for t_id, s in zip(text_ids, text_elements)
            ]
            retriever.vectorstore.add_documents(summary_texts)

        # Tables

        table_elements = list(filter(lambda x: isinstance(x, Table), parsed_doc))
        tables = list(map(lambda x: x.metadata.text_as_html, table_elements))
        table_ids = [str(uuid.UUID(bytes=hashlib.sha256(t.encode()).digest()[:16])) for t in tables]

        retriever.docstore.mset(list(zip(table_ids, tables)))

        retriever.vectorstore.embeddings
        if needs_table_summary:
            print("Saving table summaries...")
            summary_tables = [
                Document(page_content=summarize_table(s, summarizer_model, t_id), metadata={id_key: t_id})
                for t_id, s in zip(table_ids, tables)
            ]
            retriever.vectorstore.add_documents(summary_tables)
    elif isinstance(vectorstore, VectorStore):
        # Text
        text_elements = list(filter(lambda x: isinstance(x, Text), parsed_doc))
        texts = []
        text_ids = []  # [str(uuid.UUID(bytes=hashlib.sha256(t.encode()).digest()[:16])) for t in texts]
        text_metadatas = []
        for txt_elem in text_elements:
            txt_id = txt_elem.id
            if txt_id in text_ids:
                continue
            txt = summarize_text_block(txt_elem, summarizer_model, txt_id)
            m = txt_elem.metadata
            texts.append(txt)
            text_ids.append(txt_id)
            text_metadatas.append({
                "page_number": m.page_number,
                "file": m.filename,
                "type": "text block"
            })
        vectorstore.add_texts(texts, ids=text_ids, metadatas=text_metadatas)

        # Tables
        table_elements = list(filter(lambda x: isinstance(x, Table), parsed_doc))
        tables = []
        table_ids = []  # [str(uuid.UUID(bytes=hashlib.sha256(t.encode()).digest()[:16])) for t in tables]
        table_metadatas = []
        for txt, txt_id in zip(map(lambda x: x.metadata.text_as_html, table_elements), map(lambda e: e.id, table_elements)):
            if txt_id in table_ids:
                continue
            tables.append(txt)
            table_ids.append(txt_id)
            table_metadatas.append({
                "page_number": m.page_number,
                "file": m.filename,
                "type": "html table"
            })
        vectorstore.add_texts(tables, ids=table_ids, metadatas=table_metadatas)

    print("Retriever loading done")

    return retriever


def format_docs(docs: List[Document]):
    return "------\n\n".join([
        "Page number: %s\nType: %s\n%s" % (
            d.metadata.get("page_number", "Unknown"),
            d.metadata.get("type", "text"),
            d.page_content
        )
        for d in docs
    ])


def retrieval_chain(retriever: BaseRetriever, chat_model: BaseChatModel):
    template = """Fetch requested information in the following context and answer directly, without explanations and without repeating the request. If not found, say "Unknown". If field is present but blank or no value, mention "N/A".
    Example:
        Context: ```The cost was $22.4 and there were 10 employees.```
        Request: How much was the cost?
        AI: $22.4
        Request: How many employees?
        AI: 10

    Tips:
    - The context is like a letter, with a Date mentioned at the beginning of page 1. This can be used as the report date.
    - If the report is not "first report" or "preliminary", then "first report date" should be "N/A" as this can't be known.

    Start.

    Context:
    ```
    {context}
    ```
    Request (final): {question}
    AI: """
    prompt = ChatPromptTemplate.from_template(template)

    # | RunnableLambda(print_me)
    return (
        {"relevant_docs": retriever, "question": RunnablePassthrough()}
        | RunnablePassthrough.assign(context=lambda val: format_docs(val['relevant_docs']))
        | RunnableParallel(
            answer=prompt | chat_model | StrOutputParser(),
            context=RunnablePick("relevant_docs") | RunnableLambda(lambda d: d.page_content).map(),
            prompt=prompt
        )
    )
