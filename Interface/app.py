from flask import Flask, render_template, request, jsonify, Response, stream_with_context
import os
import re
from langchain_community.llms import HuggingFaceHub
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

app = Flask(__name__)

# RAG system setup

# API key and file pathss
api_key = "<hf_key>"
data_file_path = "./Data/Database_files/Sample.pdf"
embedding_data_path = "./Data/Embedding_Store/"

Exhibits = [
    {
        "title": "Ancient Egyptian Artifacts",
        "description": "Explore the rich history of Ancient Egypt.",
        "image": "./static/images/logo.jpg",  # Replace with actual image URL
        "link": "/exhibit1"
    },
    {
        "title": "Renaissance Paintings",
        "description": "Discover masterpieces from the Renaissance era.",
        "image": "./static/images/logo.jpg",  # Replace with actual image URL
        "link": "/exhibit2"
    },
    {
        "title": "Modern Art",
        "description": "Experience the evolution of modern art.",
        "image": "./static/images/logo.jpg",  # Replace with actual image URL
        "link": "/exhibit3"
    },
    # Add more exhibits as needed
]

detail_exhibits = [
    {
        "id": 1,
        "image": "https://example.com/image1.jpg",
        "title": "Ancient Artifact",
        "short_description": "A fascinating artifact from ancient history.",
        "long_description": "This artifact was discovered in the ruins of an ancient city...",
        "additional_info": "It dates back to 1500 BCE and was used for ceremonial purposes."
    },
    {
        "id": 2,
        "image": "https://example.com/image2.jpg",
        "title": "Modern Art Piece",
        "short_description": "A stunning example of modern art.",
        "long_description": "This artwork represents the abstract nature of human emotions...",
        "additional_info": "Created by a famous artist in 2021, this piece is part of the XYZ collection."
    }
]


# Indexing function
def index_docs():
    loader = PyPDFLoader(data_file_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    docs = loader.load()
    splits = text_splitter.split_documents(docs)
    
    vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)
    return vectorstore

# Set up vectorstore
if os.path.exists(embedding_data_path):
    vectorstore = Chroma(persist_directory=embedding_data_path, embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"))
else:
    vectorstore = index_docs()

# Load the LLM
def load_llm():
    llm = HuggingFaceHub(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        task="text-generation",
        model_kwargs={
            "max_new_tokens": 512,
            "top_k": 30,
            "temperature": 0.1,
            "repetition_penalty": 1.03
        },
        huggingfacehub_api_token=api_key
    )
    return llm

# Create RAG Chain
def create_rag_chain():
    retriever = vectorstore.as_retriever()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    prompt = hub.pull("rlm/rag-prompt")
    
    rag_chain = (
        {"context": retriever | format_docs,
         "question": RunnablePassthrough()}
        | prompt
        | load_llm()
        | StrOutputParser()
    )
    return rag_chain

# Create RAG chain instance
rag_chain = create_rag_chain()

# Flask routes
@app.route('/')
def home():
    return render_template('home.html', active_page='home')

def get_exhibit_by_id(exhibit_id):
    for exhibit in detail_exhibits:
        if exhibit['id'] == exhibit_id:
            return exhibit
    return None  # Return None if the exhibit is not found

# Route to display an exhibit by its ID
@app.route('/exhibit/<int:exhibit_id>')
def exhibit_page(exhibit_id):
    exhibit = get_exhibit_by_id(exhibit_id)
    if exhibit:
        return render_template('exhibit.html', exhibit=exhibit)
    
@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html', active_page='chatbot')

# Chatbot query route that integrates the RAG system
@app.route('/get', methods=['POST'])
def query():
    user_input = request.form["msg"]

    # Create a generator function for streaming
    def generate():
        # You can modify this section to simulate partial responses if necessary
        output = rag_chain.invoke(user_input)

        # Regex pattern to find the answer in the output
        pattern = r"Answer:\s*(.*?)(?:\s*Question:|$)"
        result = re.search(pattern, output, re.DOTALL)

        if result:
            response = result.group(1).strip()  # Extract and clean the answer
            yield response  # Stream the answer
        else:
            yield "Sorry, I don't know the answer to that."

    # Return a streaming response
    return Response(stream_with_context(generate()), content_type='text/event-stream')

@app.route('/about')
def about():
    return render_template('about.html', exhibits=exhibits, active_page='about')

@app.route('/exhibits')
def exhibits():
    return render_template('exhibits.html', exhibits=Exhibits, active_page='exhibits')



if __name__ == '__main__':
    app.run(debug=True)
