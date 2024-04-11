import argparse
import os
import traceback
from typing import Type
import pandas as pd

import yaml

from .rag_test_base import BaseTestRAG

from langchain_core.language_models import BaseChatModel
from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.chat_models.openai import ChatOpenAI


def get_chat_model():
    return ChatOllama(model="gemma:2b", base_url="http://hercules.local:11434")


def split_str_tuple(iter):
    for i in iter:
        if isinstance(i, str):
            yield i, None
        elif isinstance(i, tuple):
            yield i
        else:
            print('Warning: Unknown item type "%s" found. Skipping.' % type(i))


def get_tests(
        document_config_file: os.PathLike,
        rag_test_class: Type[BaseTestRAG],
        model: BaseChatModel):
    cfg = {}
    with open(document_config_file) as f:
        cfg = yaml.safe_load(f)

    rag = rag_test_class(
        os.path.join(os.path.dirname(document_config_file), cfg["document"]),
        model=model
    )
    prompts: dict = cfg["prompts"]

    def _validate_result(k_id, prompt, result):
        return {
            "id": k_id,
            "question": prompt["prompt"],
            "ground_truths": prompt.get("expected", []),
            **result,
        }

    return map(
        lambda k, p: _validate_result(k, p, rag(p["prompt"])),
        prompts.keys(),
        prompts.values(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="+")
    parser.add_argument("-o", "--result", default="results.xlsx")
    args = parser.parse_args()

    # from .test_sift_chain import TestLangchainSift
    from .test_rag_chain_retrieval import TestLangchainRAG
    test_class = TestLangchainRAG
    test_model = get_chat_model()

    with pd.ExcelWriter(args.result, mode='w') as writer:
        for c in args.config:
            print("Testing %s with %s..." % (c, test_class))
            try:
                results: pd.DataFrame = pd.DataFrame(get_tests(c, test_class, test_model)).set_index("id")
                results.to_excel(writer, sheet_name=os.path.basename(c))
            except Exception as ex:
                print("Exception while testing:", str(ex))
                traceback.print_exc()
