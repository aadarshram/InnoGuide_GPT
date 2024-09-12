import getpass
import os
from langchain_anthropic import ChatAnthropic
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_code.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, StateGraph, START
from typing import List

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

    # Define the prompt
    prompt = PromptTemplate(
        template="""system 
        You are a grader assessing relevance of a retrieved document to a user question. If the document contains keywords related to the user question, grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
        Give a binary score 'yes' or 'no' to indicate whether the document is relevant to the question.
        Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
        user
        Here is the retrieved document: \n\n {document} \n\n
        Here is the user question: {question} \n assistant""",
    input_variables=["question", "document"],
    )

    # Define the retriever grader which grades the relevance of the retrieved document
    retrieval_grader = prompt | llm | JsonOutputParser()

    # Question router based on the relevance score
    prompt = PromptTemplate(

        template="""system You are an expert at routing a user question to a vectorstore or web search. Use the vectorstore for questions on the Constitution of India, Ministry of India or anything related to the Government of India. You do not need to be stringent with the keywords in the question related to these topics. Otherwise, use web-search. Give a binary choice 'web_search' or 'vectorstore' based on the question. Return the a JSON with a single key 'datasource' and no preamble or explanation. Question to route: {question} assistant""",
        input_variables=["question"],
    )

    # Define question router
    question_router = prompt | llm | JsonOutputParser()

    # Define the RAG chain
    prompt = PromptTemplate(
        template="""system You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. user
        Question: {question}
        Context: {context}
        Answer: assistant""",
        input_variables=["question", "context"],
    )

    rag_chain = prompt | llm | JsonOutputParser()

    # Define the hallucination grader
    hallucination_grader = prompt | llm | JsonOutputParser()


    # Format docs
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    class GraphState(TypedDict):
        question: str
        generation: str
        doc_search: str
        documents: List[str]

    # Define workflow
    workflow = StateGraph(GraphState)
    
    workflow.add_node("Doc search", doc_search)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)

    workflow.add_conditional_edges(
        START,
        route_question,
        {
            "doc_search": "doc_search",
            "vectorstore": "retrieve",
        },
    )

    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "web_search": "web_search",
            "generate": "generate",
        },
    )
    workflow.add_edge("web_search", "generate")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "websearch",
        },
    )

    agent = workflow.compile()

    return agent, vectorstore

agent, vectorstore = initialize_chain()

# Stream function to handle the output
def query_rag_agent(question):
    responses = []
    for chunk in agent.stream(question):
        responses.append(chunk)
    return ''.join(responses)

# Clean up function
def cleanup():
    vectorstore.delete_collection()
