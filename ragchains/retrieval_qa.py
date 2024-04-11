import hashlib
import uuid
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.language_models import BaseChatModel
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.chat_models.openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from unstructured.documents.elements import Text, Table

from langchain_core.documents import Document


from .ingest import ingest_document
from .store import RetrievalStore


CACHE_DIR = "documents/"
needs_text_summary = False
needs_table_summary = False


def default_chat_model():
    return ChatOllama(model="gemma:2b")


def print_me(inputs):
    print("inputs:", inputs)
    return inputs


def summarize_text_block(text: str, chat_model: BaseChatModel, text_id: str):
    prompt = ChatPromptTemplate.from_template("The following content is an excerpt from an insurance report. Give a concise summary of the following content, retain important information. Also state the report title, if present: {data}")
    summarize_chain = {"data": RunnablePassthrough()} | prompt | chat_model | StrOutputParser()
    print("Summarizing text %s, \"%s\"..." % (text_id, text[:30]))
    return text  # summarize_chain.invoke(text)


def summarize_table(table_html: str, chat_model: BaseChatModel, table_id: str):
    prompt = ChatPromptTemplate.from_template("The following content is an excerpt from an insurance report. Give a concise summary of the following HTML table in English, retain important information: ```{data}```")
    summarize_chain = {"data": RunnablePassthrough()} | prompt | chat_model | StrOutputParser()
    print("Summarizing table %s..." % table_id)
    return summarize_chain.invoke(table_html)


def ingest_process(document_file, store: RetrievalStore, summarizer_model: BaseChatModel = None):
    print("Ingesting documents...")
    parsed_doc = ingest_document(document_file, cache_directory=CACHE_DIR)

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
                for t_id, s in zip(text_ids, texts)
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
        for txt, txt_id in zip(map(str, text_elements), map(lambda e: e.id, text_elements)):
            if txt_id in text_ids:
                continue
            texts.append(txt)
            text_ids.append(txt_id)
        # vectorstore.add_texts(texts, ids=text_ids)

        # Tables
        table_elements = list(filter(lambda x: isinstance(x, Table), parsed_doc))
        tables = []
        table_ids = []  # [str(uuid.UUID(bytes=hashlib.sha256(t.encode()).digest()[:16])) for t in tables]
        for txt, txt_id in zip(map(lambda x: x.metadata.text_as_html, table_elements), map(lambda e: e.id, table_elements)):
            if txt_id in table_ids:
                continue
            tables.append(txt)
            table_ids.append(txt_id)
        vectorstore.add_texts(tables, ids=table_ids)

    print("Retriever loading done")

    return retriever


def retrieval_chain(retriever: BaseRetriever, chat_model: BaseChatModel):
    template = """Fetch requested information in the following context and answer directly, without explanations. If not found, say "Don't know".
    Example:
        Context: ```The cost was $22.4 and there were 10 employees.```
        Request: How much was the cost?
        AI: Cost: $22.4
        Request: How many employees?
        AI: Employees: 10

    Context:
    ```
    {context}
    ```
    Request: {question}
    AI: """
    prompt = ChatPromptTemplate.from_template(template)

    return (
        {"context": retriever, "question": RunnablePassthrough()}
        # | RunnableLambda(print_me)
        | prompt
        | chat_model
        | StrOutputParser()
    )
