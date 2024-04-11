import time
from .rag_test_base import BaseTestRAG

from ragchains.store import global_store
from ragchains.retrieval_qa import ingest_process, retrieval_chain

from langchain_core.language_models import BaseChatModel


class TestLangchainRAG(BaseTestRAG):
    def __init__(self, document_file: str, model: BaseChatModel):
        self._document = document_file
        self._docs = ingest_process(document_file, store=global_store)
        self._eval_chain = retrieval_chain(self._docs, model)

    def __call__(self, query, extra_context=None):
        ts = time.time()
        print(query[:15], '...', end=': ')
        res = self._eval_chain.invoke(query)
        tt = time.time() - ts
        res.update({"time_taken": tt})
        print(res.get("answer"), "; time: %.2f s" % tt)
        return res
