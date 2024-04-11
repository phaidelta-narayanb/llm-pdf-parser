import torch
from functools import partial
from typing import List
from unstructured.documents.elements import Element
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnablePick, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# import torch
from transformers import MarkupLMProcessor, MarkupLMForQuestionAnswering

from .ingest import parse_document_unstructured

# Token codec
processor = MarkupLMProcessor.from_pretrained(
    # "microsoft/markuplm-base-finetuned-websrc",
    "microsoft/markuplm-large-finetuned-websrc",
    cache_dir="./cache/",
)
# Language Model
model = MarkupLMForQuestionAnswering.from_pretrained(
    # "microsoft/markuplm-base-finetuned-websrc",
    "microsoft/markuplm-large-finetuned-websrc",
    cache_dir="./cache/",
)

# chat_model = ChatOllama(model="gemma:2b", base_url="http://hercules.local:11434")
# chat_model = ChatOpenAI()

def print_me(inputs):
    print('Printing!!')
    print(inputs)
    return inputs


def extract_html_from_elements(elements: List[Element]):
    return [
        e.metadata.text_as_html
        if e.metadata.text_as_html is not None
        else "<p>%s</p>" % e.text
        for e in elements
    ]


def extract_answer_from_html_llm(input, chat_model) -> str:
    prompt = ChatPromptTemplate.from_template("Find relevant information requested from the context given below. If the query is not found, say 'Not found'.: ```{context}```\n\nQuery: {query}")
    summarize_chain = prompt | chat_model | StrOutputParser()
    return summarize_chain.invoke(input)


def extract_answer_from_html(input, processor, model):
    ctx = input.get("context", [])
    if isinstance(ctx, str):
        ctx = [ctx]
    elif ctx is None:
        ctx = []

    encoding = processor(
        '\n'.join(["<!DOCTYPE html><html><head></head><body>", *ctx, "</body></html>"]),
        questions=input["query"],
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**encoding)

    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax()

    predict_answer_tokens = encoding.input_ids[0, answer_start_index : answer_end_index + 1]
    return processor.decode(predict_answer_tokens, skip_special_tokens=True)


def ingest_chain():
    return (
        RunnableLambda(parse_document_unstructured) |
        RunnableLambda(extract_html_from_elements)
    )

# docs = ingest_chain.invoke(p)

query_engine = partial(extract_answer_from_html, processor=processor, model=model)
# query_engine = extract_answer_from_html_llm


def select_best_tool(query):
    return ""


def extract_chain(ingested_documents):
    def get_context(query: str):
        return ingested_documents[:5]

    return (
        # {"tool": RunnableLambda(select_best_tool), "query": RunnablePassthrough()} |
        # RunnableAssign() |
        {"context": get_context, "query": RunnablePassthrough()}
        | RunnableParallel(
            answer=RunnableLambda(query_engine) | StrOutputParser() | RunnableLambda(str.strip),
            context=RunnablePick("context") | RunnableLambda(str).map()
        )
    )


def run(q):
    print(q, end=': ')
    r = extract_chain.invoke(q)
    print(r)


# data_needed = [
#     "Type of loss (eg. Fire, Hurricane, Products Liability)",
#     "Secondary Type of loss (eg. Fire by arson or forest fire, Products Liability by self damage, None, Unknown, etc.)",
#     "Report Type (Eg. Preliminary, First, Final)",
#     "Report Title",
#     "Report Date",
#     "File Number",
#     "Loss Date",
#     "First Contact Date",
#     "Site Visit Date",
#     "First Report Date",
#     "Estimate",
#     "Final Claim Paid",
#     "Date Final Settlement Agreed",
#     "Date Claim Closed",
#     "Date Of First Payment",
#     # "Cause Of Loss Level1",
#     # "Cause Of Loss Level2",
#     # "Instruction Date",
#     # "Broker Name",
#     # "Threshold Payments Completed Date",
# ]

# list(map(run, data_needed))
