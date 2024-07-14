from typing import List
from common.utils import make_default_prompt_template
from question_gen.types import SubQuestionExample, ParaphrasedQuestion

# sub question decomposition
SUB_QUESTION_DEFAULT_EXAMPLES = [
    SubQuestionExample.from_default(
        question="What's the difference between LangChain agents and LangGraph?",
        sub_questions=[
            "What's the difference between LangChain agents and LangGraph?",
            "What are LangChain agents",
            "What is LangGraph"
        ]
    ),
    SubQuestionExample.from_default(
        question="How to build multi-agent system and stream intermediate steps from it",
        sub_questions=[
            "How to build multi-agent system",
            "How to stream intermediate steps",
            "How to stream intermediate steps from multi-agent system"
        ]
    ),
    SubQuestionExample.from_default(
        question="How would I use LangGraph to build an automaton",
        sub_questions=["How to build automaton with LangGraph"]
    )
]


def serialize_examples(examples: List[SubQuestionExample]):
    example_str = ""
    for i, example in enumerate(examples):
        example_str += (str(example) % (i + 1))
    return example_str


SUB_QUESTION_SYSTEM_PROMPT = f"""
Given a user question, output a list of relevant sub-questions \
in json markdown that when composed can help answer the full user question: \

Here are some example:
{serialize_examples(SUB_QUESTION_DEFAULT_EXAMPLES)}
These are the guidelines you consider when completing your task:
* Each sub question should be about a single and specific concept/fact/idea.
* The sub questions should be relevant to the user question
* The sub questions should be answerable
"""

DEFAULT_SUB_QUESTION_PROMPT_TMPL = make_default_prompt_template(
    system_prompt=SUB_QUESTION_SYSTEM_PROMPT
)


# query paraphrase
PARAPHRASED_QUESTION_JSON_SCHEMA = ParaphrasedQuestion.schema_json()

PARAPHRASED_QUESTION_SYSTEM_PROMPT = f"""You are an expert at paraphrasing user questions. \

Perform query expansion. If there are multiple common ways of phrasing a user question \
or common synonyms for key words in the question, make sure to return multiple versions \
of the query with the different phrasings.

If there are acronyms or words you are not familiar with, do not try to rephrase them.
Each paraphrased question should be in the following json schema:
{PARAPHRASED_QUESTION_JSON_SCHEMA}
Return at least 3 versions of the question.
"""

DEFAULT_PARAPHRASED_QUESTION_PROMPT_TMPL = make_default_prompt_template(
    system_prompt=PARAPHRASED_QUESTION_SYSTEM_PROMPT
)
