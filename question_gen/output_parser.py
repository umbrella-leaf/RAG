from typing import Any
from common.types import BaseOutputParser
from common.utils import parse_json_markdown
from question_gen.types import SubQuestion, ParaphrasedQuestion


class SubQuestionOutputParser(BaseOutputParser):

    def parse(self, output: str) -> Any:
        sub_questions = parse_json_markdown(output)
        sub_questions = [SubQuestion.parse_obj(item) for item in sub_questions]
        return sub_questions


class ParaphrasedQuestionOutputParser(BaseOutputParser):

    def parse(self, output: str) -> Any:
        paraphrased_questions = parse_json_markdown(output)
        paraphrased_questions = [ParaphrasedQuestion.parse_obj(item) for item in paraphrased_questions]
        return paraphrased_questions
