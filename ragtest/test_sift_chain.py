import time
from .rag_test_base import BaseTestRAG

from ragchains.sift_info import extract_chain, ingest_chain


class TestLangchainSift(BaseTestRAG):
    def __init__(self, document_file: str, **kwargs):
        self._document = document_file
        self._docs = ingest_chain().invoke(self._document)
        self._eval_chain = extract_chain(self._docs)

    def __call__(self, query, extra_context=None):
        ts = time.time()
        print(query[:15], '...', end=': ')
        res = self._eval_chain.invoke(query)
        tt = time.time() - ts
        res.update({"time_taken": tt})
        print(res.get("answer"), "; time: %.2f s" % tt)
        return res
