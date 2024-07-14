from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = PyPDFLoader("docs/sample2.pdf")
text = loader.load()

text_spliter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=20)
text_chunks = text_spliter.split_documents(text)

for text_chunk in text_chunks:
    print(text_chunk.page_content)
    print(text_chunk.metadata)

# embed_model = HuggingFaceEmbeddings(
#     model_name="nomic-ai/nomic-embed-text-v1",
#     cache_folder="embed_model",
#     model_kwargs={
#         "trust_remote_code": True,
#         "local_files_only": True,
#         "model_kwargs": {
#             "rotary_scaling_factor": 2
#         },
#         "tokenizer_kwargs": {
#             "model_max_length": 8192
#         }
#     }
# )

