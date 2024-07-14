import os

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings

embed_model = HuggingFaceEmbeddings(
    model_name="nomic-ai/nomic-embed-text-v1",
    cache_folder="embed_model",
    model_kwargs={
        "trust_remote_code": True,
        "local_files_only": True,
        "model_kwargs": {
            "rotary_scaling_factor": 2
        },
        "tokenizer_kwargs": {
            "model_max_length": 8192
        }
    }
)

os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"

embed_model1 = HuggingFaceEndpointEmbeddings(
    model="mixedbread-ai/mxbai-embed-large-v1",
)

import time
# start = time.time()
# for i in range(20):
#     embed_model.embed_query("I am Groot.")
# end = time.time()
# print(f"cost {(end - start) / 10}s to inference")

start = time.time()
for i in range(20):
    embed_model1.embed_query("He is a very goog and active man.")
end = time.time()
print(f"cost {(end - start) / 10}s to inference")

