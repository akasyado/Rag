import os
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyMuPDFLoader
from dotenv import load_dotenv
import time

load_dotenv()

api_key = os.getenv("PINECONE_API_KEY")

pc = Pinecone(api_key=api_key)
spec = ServerlessSpec(cloud="aws",region="us-east-1")

INDEX_NAME = os.getenv("INDEX_NAME")


if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name = INDEX_NAME,
        dimension = 1024,
        metric = "cosine",
        spec = spec
    )

    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(1)

loader = PyMuPDFLoader(r"book\Hands-On Machine Learning with Scikit-Learn, Keras.pdf")
data = loader.load()

data = data[19:704]  # keeping the pages of chapters only excluding index and appendix

text_splitter = RecursiveCharacterTextSplitter(chunksize = 8000,chunk_overlap = 400)
docs = text_splitter.split_documents(data)

model_name = "Qwen/Qwen3-Embedding-0.6B"

embedding = HuggingFaceEmbeddings(
    model_name = model_name,
    model_kwargs ={"device" : "cuda"},
    encode_kwargs={
        'normalize_embeddings': True,
        'batch_size': 10
        }
    )

vectorstore = PineconeVectorStore.from_documents(
    docs,
    embedding,
    index_name=INDEX_NAME
)

print("Successfully Uploaded")
