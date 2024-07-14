import os
import re
import json
import time
from typing import List, Any
from langchain_core.output_parsers import StrOutputParser
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings, HuggingFaceEmbeddings
from question_gen.generators import SubQuestionGenerator, ParaphrasedQuestionGenerator
from query_transform.transforms import HyDeQueryTransform



os.environ["NVIDIA_API_KEY"] = "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"

llm = ChatNVIDIA(model="mistralai/mixtral-8x7b-instruct-v0.1")
#
# import time
#
# question_generator = LLMQuestionGenerator.from_defaults(
#     llm=llm
# )
#
# start = time.time()
#
# print(question_generator.generate("Describe the trade-offs between using BFloat16 as the computation data type and other possible choices. When would you choose one over the other?"))
#
# end = time.time()
#
# print(f"response time: {end - start}s")

question_generator = ParaphrasedQuestionGenerator.from_defaults(
    llm=llm
)

transform = HyDeQueryTransform.from_defaults(
    llm=llm
)

start = time.time()
print(transform("how to use multi-modal models in a chain and turn chain into a rest api"))
end = time.time()
print(f"response time: {end - start}s")
