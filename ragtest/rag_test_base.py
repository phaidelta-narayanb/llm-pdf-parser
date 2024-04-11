
from typing import Optional
import os

from langchain_core.language_models import BaseChatModel


class BaseTestRAG:
    def __init__(self, document_file: os.PathLike, model: BaseChatModel = None):
        pass

    def __call__(self, query: str, extra_context: Optional[str] = None):
        return {"answer": "Not found", "time_taken": 0.0}
