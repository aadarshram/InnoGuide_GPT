import getpass
import os
from langchain_anthropic import ChatAnthropic
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_code.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Set API key
os.environ['ANTHROPIC+API_KEY'] = getpass.getpass()

# Select LLM
llm = ChatAnthropic(model="claude-3-5-sonnet-20240620")

def initialize_chain():
    # Load, chunk, and index the contents of the data
    file_name = "constitution_of_india.pdf"
    data_file_path = f"./Data/Database_files/{file_name}"
    loader = PyPDFLoader(data_file_path)
    docs = loader.load()

    # Split and chunk the text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
    splits = text_splitter.split_documents(docs)

    # Store the chunks in a vectorstore
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

    # Set up retriever
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    prompt = hub.pull("rlm/rag-prompt")

    # Format docs
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Define the RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain, vectorstore

rag_chain, vectorstore = initialize_chain()

# Stream function to handle the output
def query_rag_chain(question):
    responses = []
    for chunk in rag_chain.stream(question):
        responses.append(chunk)
    return ''.join(responses)

# Clean up function
def cleanup():
    vectorstore.delete_collection()
