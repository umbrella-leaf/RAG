from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine, SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core import VectorStoreIndex, SummaryIndex, SimpleKeywordTableIndex
from abc import ABC

vector_query_engine = VectorStoreIndex().as_query_engine()
query_engine_tools = [
    QueryEngineTool(
        query_engine=vector_query_engine,
        metadata=ToolMetadata(
            name="qlora_paper",
            description="Efficient Finetuning of Quantized LLMs",
        ),
    ),
]



query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=query_engine_tools,
    service_context=service_context,
    use_async=True,
)

vector_index = VectorStoreIndex(
    nodes, service_context=service_context
)

# summary index
summary_index = SummaryIndex(
    nodes, service_context=service_context
    )

# keyword index
keyword_index = SimpleKeywordTableIndex(nodes, service_context=service_context)

summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    service_context=service_context
)

vector_query_engine = vector_index.as_query_engine(service_context=service_context)

keyword_query_engine = keyword_index.as_query_engine(service_context=service_context)

from llama_index.core.tools.query_engine import QueryEngineTool

summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_query_engine,
    description=(
        "Useful for summarization questions related to Efficient Finetuning QLORA reserach paper"
    ),
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description=(
        "Useful for retrieving specific context from QLORA reserach paper related to Efficient Finetuning "
    ),
)

keyword_tool = QueryEngineTool.from_defaults(
    query_engine=keyword_query_engine,
    description=(
        "Useful for retrieving specific context from QLORA reserach paper related to Efficient Finetuning "
        "using entities mentioned in query"
    ),
)

from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors.llm_selectors import LLMSingleSelector, LLMMultiSelector

router_query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(service_context=service_context),
    query_engine_tools=[
        summary_tool,
        vector_tool,
        keyword_tool,
    ],
    service_context=service_context,
)

response = router_query_engine.query("What is Double Quantization?")

