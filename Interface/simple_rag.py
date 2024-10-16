'''
Building a RAG system

We use Langchain Framework to build a RAG system
Langchain is an open-source framework that provides some high-level abstraction in building applications with LLMs

'''

# There are two main components:
# Indexing: Pipeline for ingesting the data and indexing it (offline)
# Retrieval and Generation: Pipeline for retrieving the relevant data and generating the output (online)

# 1. Indexing
# Load the data using Document Loaders
# Split and chunk the data using Text Splitters
# Convert to embeddings and store using Embeddings model and VectorStore

# 2. Retrieval and Generation
# Given user query, retrieve relevant stuff from storage using Retriever
# Prompt an LLM with the user query and retrieved data for more context-aware generation


# Import necessary libraries
import os
import re
from flask import Flask, request, render_template
from langchain_community.llms import HuggingFaceHub
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Api
api_key = "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"

# Path for data file
data_file_path = "./Data/Database_files/Sample.pdf"
# Path to store Embedding data
embedding_data_path = "./Data/Embedding_Store/"

# Set up the instances of loader, text_splitter and embedding_model
loader = PyPDFLoader(data_file_path) #, extract_images = True)
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200) # We break into chunks of size 1000 and also ensure an overlap of size 200 to maintain continuity
embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2") # We use an open-source Embeddings model from HF to embed into vectors

def index_docs():
    # Load text from PDF - We use PyPDF for this. Additionally, we use rapidocr for extracting useful text from images too.
    docs = loader.load()
    # Chunk
    # We use RecursiveCharacterTextSplitter to split the text into chunks recursively until chunk size is reached
    # Other alternatives may be explored
    splits = text_splitter.split_documents(docs)
    # Convert to Embeddings
    # We use open-source vectorstore Chroma for vector storage (Alternatively, we can use Faiss as well)
    vectorstore = Chroma.from_documents(documents = splits, embedding = embedding_model)  

    return vectorstore

# Set vectorstore instance
if os.path.exists(embedding_data_path):
    # Load existing vectorstore
    vectorstore = Chroma(persist_directory = embedding_data_path, embedding_function = embedding_model)
else:
    # Create new 
    vectorstore = index_docs()

# Set up the LLM - We use an open-source llm from HF
def load_llm():
    llm = HuggingFaceHub(
        repo_id = "HuggingFaceH4/zephyr-7b-beta",
        task = "text-generation",
        model_kwargs = {
            "max_new_tokens": 512,
            "top_k": 30,
            "temperature": 0.1,
            "repetition_penalty": 1.03
        },
        huggingfacehub_api_token = api_key
    )
    return llm

# Load an instance of LLM
llm = load_llm()

# Define the RAG chain
def create_rag_chain():
    # Set retriever
    retriever = vectorstore.as_retriever() # Internally uses vector semantic search for retrieval

    def format_docs(docs):
        '''
        Function to format a page-wise doc to single one
        '''
        return "\n\n".join(doc.page_content for doc in docs)

    prompt = hub.pull("rlm/rag-prompt") # Standard prompt for RAG system


    rag_chain = ( # Create a RAG chain
        {"context": retriever | format_docs, # Retrieve relevant docs and format them
        "question": RunnablePassthrough()} # User query 
        | prompt # System prompt to RAG
        | llm # LLM to use
        | StrOutputParser() # Parse the resultant output to get onlythe string response
    )

    return rag_chain

# Create an instance of the RAG chain
rag_chain = create_rag_chain()


@app.route('/chat', methods=['POST'])
def chat():
    query = request.form['user_input']
    output = rag_chain.invoke(query)
    # Extract only the answer
    pattern = r"Answer:\s*(.*)"
    output = re.search(pattern, output, re.DOTALL)
    if output:
        output = output.group(1).strip()
        return {'response': output}
    else:
        return {"response": "Sorry, I dont know"}

# Clean up 
# vectorstore.delete_collection()
