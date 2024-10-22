import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain import hub
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.prebuilt import create_react_agent
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


def initialize_chain():

    memory = MemorySaver(
    )    
    embeddings = OpenAIEmbeddings(model = "text-embedding-3-large" , api_key = openai_api_key)
    try:
        vectorstore = Chroma(persist_directory="Data/vectorstore", embedding_function=embeddings)
        print('Using existing vectorstore')

    except:
        print('Creating a new vectorstore')
        # Load, chunk, and index the contents of the data
        files = ["sample.pdf"]
        for file_name in files:
            data_file_path = f"./Data/Database_files/{file_name}"
            if file_name.endswith(".pdf"):
                loader = PyPDFLoader(data_file_path)
            elif file_name.endswith(".txt"):
                loader = TextLoader(data_file_path)
        docs = loader.load()

        # Split and chunk the text
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
        splits = text_splitter.split_documents(docs)

        # Store the chunks in a vectorstore
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory = "./Data/vectorstore")

      # Select LLM
    llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key=openai_api_key,
            # other params...
        )
    # Set up retriever
    retriever = vectorstore.as_retriever()

    # (Optional) Set up a multi-query retriever for better performance (check with latency)
    # retriever = MultiQueryRetriever(retriever, llm)
    # def format_docs(docs):
    #     return "\n\n".join(doc.page_content for doc in docs)
    retriever_tool = create_retriever_tool(retriever, "law_content_retriever", "Searches and returns relevant excerpts from the Law and History of India document.")
  
    tools = [retriever_tool]
    # prompt = hub.pull("hwchase17/openai-tools-agent")
    # prompt_template = PromptTemplate(
    #     template="""
    # You are an intelligent assistant designed to provide accurate information and assist users with their queries. You can either use Retrieval-Augmented Generation (RAG) or respond based on your training.

    # ### Instructions:
    # 1. **Use RAG**:
    # - If the question is specific and requires detailed information from external documents.
    # - If the user asks for recent events or precise data.

    # 2. **Use Independent Response**:
    # - If the quewhat istion is general or commonly known.
    # - If you can provide a confident and concise answer without external information.

    # If you are uncertain, ask the user for clarification.

    # ### Response Format:
    # - Acknowledge the user's question.
    # - If using RAG, summarize the retrieved information before answering.
    # - If responding independently, give a clear and informative answer.
    #  Now, based on the user query: "{user_query}"
    #     Answer the user query:
    # """
    # )    
    agent_executor = create_react_agent(llm, tools, checkpointer=memory)
        
    return agent_executor, vectorstore

agent, vectorstore = initialize_chain()

# Stream function to handle the output
def query_rag_agent(question):
    responses = []
    # formatted_prompt = prompt_template.format(user_query = question)
    formatted_prompt = 'You are a helpful assistant. Use RAG tool for context. Else answer user query based on trained knowledge. User Query: ' + question
    for chunk in agent.stream(
        {"messages": [HumanMessage(content=formatted_prompt)]},
        {"configurable": {"thread_id": "1"}}
    ):
        # Assuming each chunk is a dictionary with the structure you provided
        if 'agent' in chunk and 'messages' in chunk['agent']:
            # Extract the content of the first message in the messages list
            content = chunk['agent']['messages'][0].content
            responses.append(content)
    
    # Join all pieces of content to form the final output
    return ''.join(responses)


# Clean up function
# def cleanup():
    # vectorstore.delete_collection()

if __name__ == "__main__":
    # For testing
    question = input("Enter your question: ")
    response = query_rag_agent(question)
    print(response)
    # cleanup()