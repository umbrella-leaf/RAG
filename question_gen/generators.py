from typing import Optional, Union, List
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import BasePromptTemplate, ChatPromptTemplate
from common.types import BaseOutputParser, PromptTemplate
from question_gen.types import BaseQuestionGenerator, SubQuestion, ParaphrasedQuestion
from question_gen.output_parser import SubQuestionOutputParser, ParaphrasedQuestionOutputParser
from question_gen.prompts import (
    DEFAULT_SUB_QUESTION_PROMPT_TMPL,
    DEFAULT_PARAPHRASED_QUESTION_PROMPT_TMPL
)


class SubQuestionGenerator(BaseQuestionGenerator):
    def __init__(
        self,
        llm: BaseChatModel,
        prompt: BasePromptTemplate,
        output_parser: BaseOutputParser
    ) -> None:
        self._llm = llm | StrOutputParser()
        self._prompt = prompt
        self._output_parser = output_parser
        self._chain = (self._prompt | self._llm | self._output_parser)

    @classmethod
    def from_defaults(
            cls,
            llm: BaseChatModel,
            prompt_template: Optional[PromptTemplate] = None,
            output_parser: Optional[BaseOutputParser] = None
    ) -> 'SubQuestionGenerator':
        prompt = prompt_template if isinstance(prompt_template, BasePromptTemplate) \
                 else ChatPromptTemplate.from_template(prompt_template) if isinstance(prompt_template, str) \
                 else DEFAULT_SUB_QUESTION_PROMPT_TMPL
        output_parser = output_parser or SubQuestionOutputParser()
        return cls(llm, prompt, output_parser)

    def generate(
        self, query: str
    ) -> List[SubQuestion]:
        return self._chain.invoke({
            "question": query
        })

    async def agenerate(
            self, query: str
    ) -> List[SubQuestion]:
        return await self._chain.ainvoke({
            "question": query
        })


class ParaphrasedQuestionGenerator(BaseQuestionGenerator):
    def __init__(
        self,
        llm: BaseChatModel,
        prompt: BasePromptTemplate,
        output_parser: BaseOutputParser
    ) -> None:
        self._llm = llm | StrOutputParser()
        self._prompt = prompt
        self._output_parser = output_parser
        self._chain = (self._prompt | self._llm | self._output_parser)

    @classmethod
    def from_defaults(
            cls,
            llm: BaseChatModel,
            prompt_template: Optional[PromptTemplate] = None,
            output_parser: Optional[BaseOutputParser] = None
    ) -> 'ParaphrasedQuestionGenerator':
        prompt = prompt_template if isinstance(prompt_template, BasePromptTemplate) \
                 else ChatPromptTemplate.from_template(prompt_template) if isinstance(prompt_template, str) \
                 else DEFAULT_PARAPHRASED_QUESTION_PROMPT_TMPL
        output_parser = output_parser or ParaphrasedQuestionOutputParser()
        return cls(llm, prompt, output_parser)

    def generate(
        self, query: str
    ) -> List[ParaphrasedQuestion]:
        return self._chain.invoke({
            "question": query
        })

    async def agenerate(
            self, query: str
    ) -> List[ParaphrasedQuestion]:
        return await self._chain.ainvoke({
            "question": query
        })
