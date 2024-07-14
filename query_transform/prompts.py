from common.utils import make_default_prompt_template


# HyDE query
HYDE_QUERY_SYSTEM_PROMPT = (
    "Please write a passage to answer user question\n"
    "Try to include as many key details as possible.\n"
)


DEFAULT_HYDE_QUERY_PROMPT_TMPL = make_default_prompt_template(
    system_prompt=HYDE_QUERY_SYSTEM_PROMPT
)


