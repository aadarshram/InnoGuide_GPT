import os
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.schema import Document
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict
from typing import List

# Initialize FastEmbed model
embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

# Load documents from the Data directory
def load_documents(directory="Data"):
    docs = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(directory, filename))
            docs.extend(loader.load())
        elif filename.endswith(".txt"):
            loader = TextLoader(os.path.join(directory, filename))
            docs.extend(loader.load())
    return docs

# Split documents into manageable chunks
def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512, chunk_overlap=0
    )
    return text_splitter.split_documents(docs)

# Initialize the vector store
def initialize_vector_store(doc_splits):
    return Chroma.from_documents(documents=doc_splits, embedding=embed_model, collection_name="local-rag")

# Set up prompts
def setup_prompts():
    question_router = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a 
        user question to a vectorstore or web search. Use the vectorstore for questions on LLM agents, 
        prompt engineering, and adversarial attacks. Otherwise, use web-search. Return 'web_search' or 'vectorstore' in a JSON with a key 'datasource'. Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question"],
    )

    answer_chain = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
        Question: {question} 
        Context: {context} 
        Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question", "context"],
    )
    
    return question_router, answer_chain

# Define the state
class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    web_search: str

# Function to retrieve documents
def retrieve(state, retriever):
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

# Function to generate an answer
def generate(state, rag_chain):
    question = state["question"]
    documents = state["documents"]
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

# Define and compile the workflow
def compile_workflow(retriever, rag_chain):
    workflow = StateGraph(GraphState)
    
    workflow.add_node("retrieve", lambda state: retrieve(state, retriever))
    workflow.add_node("generate", lambda state: generate(state, rag_chain))

    # Define routing logic
    def route_question(state):
        question = state["question"]
        source = question_router.invoke({"question": question})  
        return "websearch" if source['datasource'] == 'web_search' else "retrieve"

    workflow.set_conditional_entry_point(route_question, {"websearch": "websearch", "vectorstore": "retrieve"})
    workflow.add_edge("retrieve", "generate")

    app = workflow.compile()
    return app

def main():
    # Load documents and initialize vector store
    docs = load_documents()
    doc_splits = split_documents(docs)
    vectorstore = initialize_vector_store(doc_splits)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    # Set up prompts
    global question_router, rag_chain
    question_router, rag_chain = setup_prompts()

    # Compile workflow
    app = compile_workflow(retriever, rag_chain)

    # Query loop
    while True:
        question = input("Enter your question (or 'exit' to quit): ")
        if question.lower() == 'exit':
            break

        # Execute workflow
        state = {"question": question, "documents": [], "generation": "", "web_search": ""}
        result = app.invoke(state)

        # Print the output
        print("Answer:", result['generation'])

if __name__ == "__main__":
    main()
