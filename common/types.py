from abc import ABC, abstractmethod
from typing import Any, Dict, Union
from langchain_core.prompts import BasePromptTemplate

PromptTemplate = Union[BasePromptTemplate, str]


class BaseOutputParser(ABC):
    @classmethod
    def __modify_schema__(cls, schema: Dict[str, Any]) -> None:
        """Avoids serialization issues."""
        schema.update(type="object", default={})

    @abstractmethod
    def parse(self, output: str) -> Any:
        """Parse, validate, and correct errors programmatically."""

    def __call__(self, output: str):
        return self.parse(output)
