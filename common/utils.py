import re
import json
from typing import Any, Optional
from langchain_core.prompts import ChatPromptTemplate


json_parse_regex: re.Pattern = re.compile('```json(.*?)```', re.DOTALL)


def parse_json_markdown(text: str) -> Any:
    if "```json" in text:
        json_str = json_parse_regex.match(text).group(1)
    else:
        json_str = text
    json_dict = json.loads(json_str.strip())
    if "items" in json_dict:
        json_dict = json_dict["items"]
    return json_dict


def make_default_prompt_template(
        system_prompt: str,
        human_prompt: Optional[str] = None
) -> ChatPromptTemplate:
    human_prompt = human_prompt or "Here is the user question: {question}"
    return ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt.replace("{", "{{").replace("}", "}}")),
            ("human", human_prompt)
        ]
    )
