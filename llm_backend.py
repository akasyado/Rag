from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from dotenv import load_dotenv
import os
from retriever import retriever

load_dotenv()

llm = ChatGroq(
    model = "llama-3.1-8b-instant",
    temperature = 0
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer ONLY from the provided context. If not found, say you don't know"),
    ("human", "Question: {question}\n\nContext:\n{context}")
])


parallel = RunnableParallel({
    "context" : retriever,
    "question" : RunnablePassthrough()
})

chain = parallel | prompt | llm | StrOutputParser()


if __name__ == "__main__":
    try:
        while True:
            print("PDF RAG ready. Ask a question (or ctrl + c to exit).")
            q = input("\nQ:")
            ans = chain.invoke(q.strip())
            print("\nA:", ans)
    except KeyboardInterrupt:
        print("\n\nQuiting.......Goodbye")


