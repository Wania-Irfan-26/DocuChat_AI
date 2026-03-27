from dotenv import load_dotenv
import os
load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import TextLoader, PyPDFLoader


#Document loading
def load_documents(file_path):
    if file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    elif file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        raise ValueError("Unsupported file type")

    return loader.load()

data_from_doc = load_documents("Tept_outlines.pdf")  # or "file.pdf"

#making small chunks of the loaded text
chunks = RecursiveCharacterTextSplitter(
    chunk_size= 100,
    chunk_overlap= 20
)

chunked_data = chunks.split_documents(data_from_doc)

#converting chunks into vectors(numbers)
embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#storing vectors in vector database(chroma)
vector_db_storage = Chroma.from_documents(
    documents= chunked_data,
    embedding = embed
)

#retrieving data(search tool now) - replaces similarity_search (manual)
retrieved_data = vector_db_storage.as_retriever()

#connectng llm
my_llm = ChatGroq(model="llama-3.1-8b-instant")


#prompt template
prompt = ChatPromptTemplate.from_template("""
Answer the question based on the context below.
Context: {context}
Question: {question}
""")

#retrieval chain
chain = (
    {"context": retrieved_data, "question": RunnablePassthrough()}
    | prompt
    | my_llm
    | StrOutputParser()
)

#taking user query
user = input("Enter your query: ")

#results
result = chain.invoke(user)
print("============Final Answer==========")
print(result)