from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

index_name = os.getenv("INDEX_NAME")

api_key=os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=api_key)


myindex = pc.Index(index_name)



model_name = "Qwen/Qwen3-Embedding-0.6B"

embed_model = HuggingFaceEmbeddings(
    model_name = model_name,
    encode_kwargs={'normalize_embeddings': True}
    )

vectorstore = PineconeVectorStore(myindex,
                                  embedding=embed_model,
                                  text_key="text"
                                  )


def retriever(query : str, k=5):
    result = vectorstore.similarity_search(query,k=k)
    result = "\n\n".join(d.page_content for d in result)
    return result
