import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

def initialize_chain():
    memory = MemorySaver()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key)
    
    # Try to load existing vectorstore or create a new one
    try:
        vectorstore = Chroma(persist_directory="Data/vectorstore", embedding_function=embeddings)
        print('Using existing vectorstore')
    except Exception as e:
        print('Creating a new vectorstore:', e)
        docs = []
        files = ["sample.pdf"]  # Add more file names as needed
        for file_name in files:
            data_file_path = f"./Data/Database_files/{file_name}"
            if file_name.endswith(".pdf"):
                loader = PyPDFLoader(data_file_path)
            elif file_name.endswith(".txt"):
                loader = TextLoader(data_file_path)
            docs.extend(loader.load())
        
        # Split and chunk the text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        # Store the chunks in a vectorstore
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./Data/vectorstore")

    # Set up the language model
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=openai_api_key
    )

    return llm, vectorstore

llm, vectorstore = initialize_chain()

def query_rag_chain(question):
    # Retrieve relevant documents
    retriever = vectorstore.as_retriever()
    retrieved_docs = retriever.invoke(question)

    # Prepare the context for the LLM
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    prompt = f"You are a helpful assistant. Use the following context to answer the question:\n\n{context}\n\nUser Query: {question}"

    # Query the language model
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return response.content

if __name__ == "__main__":
    # Initialize the chain

    # Input loop for user questions
    while True:
        question = input("Enter your question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break
        response = query_rag_agent(question)
        print("Response:", response)
