from functools import partial
import hashlib
import os
import uuid
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.chat_models.openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import Agent

from unstructured.documents.elements import Text, Table

from langchain_core.documents import Document

from bs4 import BeautifulSoup

import csv
from pathlib import Path
from difflib import SequenceMatcher

import argparse

from ingest import get_retriever, ingest_document
from store import id_key


needs_text_summary = False
needs_table_summary = False
CACHE_DIR = "documents/"

parser = argparse.ArgumentParser()
parser.add_argument("documents", nargs="+")

args = parser.parse_args()

SCAN_DOCS = args.documents


def print_docs_stats(docs):
    category_counts = {}

    for doc_elements in docs:
        for element in doc_elements:
            category = str(type(element))
            if category in category_counts:
                category_counts[category] += 1
            else:
                category_counts[category] = 1

    print(category_counts)


def summarize_text_block(text: str, chat_model: ChatOllama, text_id: str):
    prompt = ChatPromptTemplate.from_template("Give a concise summary of the following content, retain important information: {data}")
    summarize_chain = {"data": RunnablePassthrough()} | prompt | chat_model | StrOutputParser()
    print("Summarizing text %s, \"%s\"..." % (text_id, text[:30]))
    return summarize_chain.invoke(text)


def summarize_table(table_html: str, chat_model: ChatOllama, table_id: str):
    prompt = ChatPromptTemplate.from_template("Give a concise summary of the following HTML table in English, retain important information: ```{data}```")
    summarize_chain = {"data": RunnablePassthrough()} | prompt | chat_model | StrOutputParser()
    print("Summarizing table %s..." % table_id)
    return summarize_chain.invoke(table_html)


os.environ["TABLE_IMAGE_CROP_PAD"] = "96"
os.environ["EXTRACT_IMAGE_BLOCK_CROP_HORIZONTAL_PAD"] = "96"


chat_model = ChatOllama(model="codellama", base_url="http://hercules.local:11434")
summarizer_model = chat_model

print("Ingesting documents...")
parsed_docs = list(
    map(partial(ingest_document, cache_directory=CACHE_DIR), SCAN_DOCS)
)

# print_docs_stats(parsed_docs)

print("Creating retriever...")
retriever = get_retriever()


text_elements = list(filter(lambda x: isinstance(x, Text), parsed_docs[0]))
texts = list(map(str, text_elements))
text_ids = [str(uuid.UUID(bytes=hashlib.sha256(t.encode()).digest()[:16])) for t in texts]

# Store original text copy
retriever.docstore.mset(list(zip(text_ids, texts)))


if needs_text_summary:
    summary_texts = [
        Document(page_content=summarize_text_block(s, summarizer_model, t_id), metadata={id_key: t_id})
        for t_id, s in zip(text_ids, texts)
    ]
    retriever.vectorstore.add_documents(summary_texts)

print("Stored %d text blocks in docstore" % len(text_ids))


table_elements = list(filter(lambda x: isinstance(x, Table), parsed_docs[0]))
tables = list(map(lambda x: x.metadata.text_as_html, table_elements))
table_ids = [str(uuid.UUID(bytes=hashlib.sha256(t.encode()).digest()[:16])) for t in tables]

retriever.docstore.mset(list(zip(table_ids, tables)))

if needs_table_summary:
    summary_tables = [
        Document(page_content=summarize_table(s, summarizer_model, t_id), metadata={id_key: t_id})
        for t_id, s in zip(table_ids, tables)
    ]
    retriever.vectorstore.add_documents(summary_tables)


print("Stored %d tables in docstore" % len(table_ids))

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
"""
prompt = ChatPromptTemplate.from_template(template)


chat_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | chat_model
    | StrOutputParser()
)


# agent = Agent()

def append_csv(csv_file, data: dict):
    csv_file = Path(csv_file)

    if not csv_file.exists():
        with open(csv_file, 'w', newline='') as op_csv:
            op_csv_write = csv.DictWriter(op_csv, data.keys())
            op_csv_write.writeheader()

    with open(csv_file, 'a', newline='') as op_csv:
        op_csv_append = csv.DictWriter(op_csv, data.keys())
        op_csv_append.writerow(data)


RUN_ID = uuid.uuid4().hex

print("Run id:", RUN_ID)


def call_prompt(q, expected=""):
    print("Processing prompt \"%s\"" % q)
    response = chat_chain.invoke(q).strip()
    print("> \"%s\", expected: \"%s\"" % (response.strip(), expected))

    append_csv("documents/run_%s.csv" % RUN_ID, {
        "query": q,
        "expected": expected,
        "actual": response,
        "similarity %": SequenceMatcher(None, expected, response).ratio()
    })



# Mclarens Liability Preliminary original

# call_prompt("What is the Date Of Loss?")  #  May 16, 2022
# call_prompt("Who is insured?") #  Shenzhen Jiumo Technology Co., Limited


# Sample document
# call_prompt("What is the Birth date?")
# call_prompt("What is the Start date?")
# call_prompt("What is the End date?")
# call_prompt("What is the Report ID?")
# call_prompt("To whom is it Addressed to?")
# call_prompt("What is the Insured item?")
# call_prompt("What is the Assured Sum?")

# - Date of Birth: 01 April 1901
# - Start date: 01 April 2011
# - End date: 01 April 2099
# - Report ID: ABCDWXYZ111
# - Addressed to: John Doe
# - Insured item: House
# - Assured Sum: $1999.99


# Original questions
# call_prompt("Who is Insured Party?", "Sample Technology Co., Limited")
# call_prompt("when is Date of Loss?", "May 30, 2022")
# call_prompt("When is Policy start date?", "October 2, 2021 ")
# call_prompt("When is Policy End date?", "October 30, 2023")
# call_prompt("return File No.", "015.0227.00")
# call_prompt("how is it addressed to?", "Nick Jackman")
# call_prompt("What is item is insured here", "Products Liability")
# call_prompt("What is the assured sum ?", "Don't know")
# call_prompt("What was original Claim amount?", "$100,000.00")
# call_prompt("Who is a claimant here?", "Nick Jackman")
# call_prompt("What is a claimant address?", "")
# call_prompt("Who is a claimant Attorney and what is thier office address?", "")
# call_prompt("What type of loss has been claimed?", "")
# call_prompt("Where did the loss occured?", "")
# call_prompt("Give Policy Number", "")
# call_prompt("what was the date of assignment", "")
# call_prompt("Give Summary of final investigation.", "")
# call_prompt("What needs to be done?", "")

# Improved prompts
# call_prompt("Who is the Insured Party?", "Sample Technology Co., Limited")
# call_prompt("When is the Date of Loss?", "May 30, 2022")
# call_prompt("When is the Policy start date?", "October 2, 2021 ")
# call_prompt("When is the Policy End date?", "October 30, 2023")
# call_prompt("What is the File No.?", "015.0227.00")
# call_prompt("Who is it addressed to?", "Nick Jackman")
# call_prompt("What is insured?", "Products Liability")
# call_prompt("What is the assured sum?", "Don't know")
# call_prompt("What is the original Claim amount?", "$100,000.00")
# call_prompt("Who is a claimant here?", "Nick Jackman")
# call_prompt("What is the claimant's address?", "")
# call_prompt("Who is the claimant's Attorney and what is thier office address?", "Jon Doe")
# call_prompt("What type of loss has been claimed?", "Products Liability")
# call_prompt("Where did the loss occur?", "4586 N Road. Panjim, Goa.")
# call_prompt("Give the Policy Number", "13740036501627517926")
# call_prompt("What was the date of assignment?", "August 1, 2022")
# call_prompt("Give Summary of final investigation.", "")
# call_prompt("What needs to be done?", "")

call_prompt("What is the report date?", "January 18, 2023")
