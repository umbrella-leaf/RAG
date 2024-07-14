from typing import Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import BasePromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from common.types import PromptTemplate
from query_transform.types import HyDeDocument, BaseQueryTransform
from query_transform.prompts import (
    DEFAULT_HYDE_QUERY_PROMPT_TMPL
)


class HyDeQueryTransform(BaseQueryTransform):
    def __init__(
        self,
        llm: BaseChatModel,
        prompt: BasePromptTemplate
    ) -> None:
        self._llm = llm | StrOutputParser()
        self._prompt = prompt
        self._chain = self._prompt | self._llm

    @classmethod
    def from_defaults(
            cls,
            llm: BaseChatModel,
            prompt_template: Optional[PromptTemplate] = None
    ) -> 'HyDeQueryTransform':
        prompt = prompt_template if isinstance(prompt_template, BasePromptTemplate) \
            else ChatPromptTemplate.from_template(prompt_template) if isinstance(prompt_template, str) \
            else DEFAULT_HYDE_QUERY_PROMPT_TMPL
        return cls(llm, prompt)

    def run(self, query: str) -> HyDeDocument:
        hypothetical_document = self._chain.invoke({
            "question": query
        })
        return HyDeDocument(
            hypothetical_document=hypothetical_document
        )
