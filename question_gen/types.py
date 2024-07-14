import json
from langchain_core.pydantic_v1 import BaseModel, Field
from abc import abstractmethod
from typing import List, Union


# sub question decomposition
class SubQuestion(BaseModel):
    sub_question: str


class SubQuestionExample(BaseModel):
    question: str
    sub_questions: List[SubQuestion]

    @classmethod
    def from_default(cls,
                     question: str,
                     sub_questions: List[str]
    ) -> 'SubQuestionExample':
        return cls(
            question=question,
            sub_questions=[SubQuestion(sub_question=sub_question) for sub_question in sub_questions]
        )

    def __str__(self):
        example_str = ""
        example_query_str = self.question
        example_output_str = json.dumps({"items": [x.dict() for x in self.sub_questions]}, indent=4)
        example_str += f"""# Example %s\n\n<User Question>\n{example_query_str}\n\n<Output>\n```json\n{example_output_str}\n```\n\n"""
        return example_str


# query expansion
class ParaphrasedQuestion(BaseModel):
    paraphrased_question: str = Field(
        ...,
        description="A unique paraphrasing of the original question.",
    )


GeneratedQuestion = Union[SubQuestion, ParaphrasedQuestion]


class BaseQuestionGenerator:
    @abstractmethod
    def generate(
        self, query: str
    ) -> List[GeneratedQuestion]:
        pass

    @abstractmethod
    async def agenerate(
            self, query: str
    ) -> List[GeneratedQuestion]:
        pass

    def __call__(self, query: str):
        return self.generate(query)
