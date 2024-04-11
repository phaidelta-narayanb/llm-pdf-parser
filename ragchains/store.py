import uuid
import os

from langchain_core.vectorstores import VectorStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import LocalFileStore, InMemoryStore
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document

from langchain_community.embeddings import HuggingFaceEmbeddings

CHROMA_STORE = "./chromadb/"


class RetrievalStore:
    def __init__(self, vectorstore: VectorStore = None):
        if vectorstore is None:
            # The vectorstore to use to index the child chunks
            vectorstore = Chroma(
                collection_name="documents",
                embedding_function=HuggingFaceEmbeddings(),
                persist_directory=os.path.join(CHROMA_STORE, "documents/")
            )
        self._vectorstore = vectorstore

    @property
    def vectorstore(self):
        return self._vectorstore

    @property
    def retriever(self):
        return self._vectorstore.as_retriever()


global_store = RetrievalStore()

# The storage layer for the parent documents
# store = InMemoryStore()  # ("./chromadb/store/")
# id_key = "doc_id"


# The retriever (empty to start)
# retriever = vectorstore.as_retriever()
# retriever = MultiVectorRetriever(
#     vectorstore=vectorstore,
#     docstore=store,
#     id_key=id_key,
# )


"""
# Add texts
doc_ids = [str(uuid.uuid4()) for _ in texts]
summary_texts = [
    Document(page_content=s, metadata={id_key: doc_ids[i]})
    for i, s in enumerate(text_summaries)
]
retriever.vectorstore.add_documents(summary_texts)
retriever.docstore.mset(list(zip(doc_ids, texts)))

# Add tables
table_ids = [str(uuid.uuid4()) for _ in tables]
summary_tables = [
    Document(page_content=s, metadata={id_key: table_ids[i]})
    for i, s in enumerate(table_summaries)
]
retriever.vectorstore.add_documents(summary_tables)
retriever.docstore.mset(list(zip(table_ids, tables)))
"""
