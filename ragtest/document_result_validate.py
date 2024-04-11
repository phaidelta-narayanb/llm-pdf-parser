import argparse
from itertools import islice
from difflib import SequenceMatcher
import os
import re
import traceback
from typing import Type
import pandas as pd

import importlib
import yaml

from .rag_test_base import BaseTestRAG

from langchain_core.language_models import BaseChatModel


def get_chat_model(model_path: str, **kwargs):
    if model_path is None:
        return
    module_name, cls_name, model_name = model_path.split("/", 2)

    mod = importlib.import_module('.'.join([
        "langchain_community.chat_models",
        module_name
    ]))
    cls_ = getattr(mod, cls_name, None)
    if cls_ is None:
        return

    return cls_(model=model_name, **{k: v for k, v in kwargs.items() if v is not None})


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

    def _evaluate_result(prompt, result):
        def _best_similarity(expected: list, answer: str):
            return max(
                map(
                    lambda v: SequenceMatcher(None, v, answer).ratio(),
                    expected
                ))

        return {
            "best_similarity": _best_similarity(
                prompt.get("expected", []),
                result.get("answer")
            )
        }

    def _validate_result(k_id, prompt, result):
        return {
            "id": k_id,
            "question": prompt["prompt"],
            "ground_truths": prompt.get("expected", []),
            **result,
            **_evaluate_result(prompt, result)
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
    parser.add_argument("-m", "--model", default=None, help="Example: openai/ChatOpenAI/gpt-3.5-turbo", required=False)
    parser.add_argument("-b", "--base-url", default=None, required=False)
    parser.add_argument("-r", "--rag-class", default="TestLangchainRAG", required=False)
    args = parser.parse_args()

    print("Using RAG class: %s" % args.rag_class)

    # TODO: Dict or auto-import
    if args.rag_class == "TestLangchainRAG":
        from .test_rag_chain_retrieval import TestLangchainRAG
        test_class = TestLangchainRAG
    elif args.rag_class == "TestLangchainSift":
        from .test_sift_chain import TestLangchainSift
        test_class = TestLangchainSift
    else:
        print("Unknown RAG class \"%s\" specified." % args.rag_class)
        exit(1)

    print("Using model: %s" % args.model)
    test_model = get_chat_model(args.model, base_url=args.base_url)

    with pd.ExcelWriter(args.result, mode='w') as writer:
        for i, c in enumerate(args.config):
            print("Testing %s with %s..." % (c, test_class))
            print("-"*15)
            try:
                sheet_name = "{model_path}_{config_idx}".format(
                    config_file=os.path.basename(c),
                    config_idx=i,
                    model_path=re.sub(r"[\[\]\:\*\?\/\\]", ".", str(args.model))
                )
                results: pd.DataFrame = pd.DataFrame(
                    islice(get_tests(c, test_class, test_model), None)
                ).set_index("id")
                results.to_excel(writer, sheet_name=sheet_name[:31])
            except Exception as ex:
                print("Exception while testing:", str(ex))
                traceback.print_exc()
            print("="*15)
