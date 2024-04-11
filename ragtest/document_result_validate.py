import argparse
import os
import traceback
from typing import Type
import pandas as pd

import importlib
import yaml

from .rag_test_base import BaseTestRAG

from langchain_core.language_models import BaseChatModel


def get_chat_model(model_path: str, **kwargs):
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
    parser.add_argument("-m", "--model", default="openai/ChatOpenAI/gpt-3.5-turbo", required=False)
    parser.add_argument("-b", "--base-url", default=None, required=False)
    args = parser.parse_args()

    # from .test_sift_chain import TestLangchainSift
    from .test_rag_chain_retrieval import TestLangchainRAG
    test_class = TestLangchainRAG

    print("Using model: %s" % args.model)
    test_model = get_chat_model(args.model, base_url=args.base_url)

    with pd.ExcelWriter(args.result, mode='w') as writer:
        for c in args.config:
            print("Testing %s with %s..." % (c, test_class))
            try:
                sheet_name = "{model_path}_{config_file}".format(
                    config_file=os.path.basename(c),
                    model_path=args.model.replace("/", "_")
                )
                results: pd.DataFrame = pd.DataFrame(get_tests(c, test_class, test_model)).set_index("id")
                results.to_excel(writer, sheet_name=sheet_name[:31])
            except Exception as ex:
                print("Exception while testing:", str(ex))
                traceback.print_exc()
