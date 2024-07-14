from typing import Union
from abc import abstractmethod
from langchain_core.pydantic_v1 import BaseModel, Field


# HyDE(Hypothetical Document Embedding) query
class HyDeDocument(BaseModel):
    hypothetical_document: str = Field(
        ...,
        description="A generated passage relevant to user question"
    )


TransformQuery = Union[HyDeDocument]


class BaseQueryTransform:
    @abstractmethod
    def run(self, query: str) -> TransformQuery:
        pass

    def __call__(self, query: str) -> TransformQuery:
        return self.run(query)
