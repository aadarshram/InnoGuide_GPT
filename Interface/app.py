from flask import Flask, render_template, request, Response, stream_with_context
from rag_agent import query_rag_agent
from rag_chain import query_rag_chain

app = Flask(__name__)


# API key and file paths
data_file_path = "./Data/Database_files/Sample.pdf"
embedding_data_path = "./Data/Embedding_Store/"

Exhibits = [
    {
        "title": "Ancient Egyptian Artifacts",
        "description": "Explore the rich history of Ancient Egypt.",
        "image": "./static/images/logo.jpg",
        "link": "/exhibit/1"
    },
    {
        "title": "Renaissance Paintings",
        "description": "Discover masterpieces from the Renaissance era.",
        "image": "./static/images/logo.jpg",
        "link": "/exhibit/2"
    },
    {
        "title": "Modern Art",
        "description": "Experience the evolution of modern art.",
        "image": "./static/images/logo.jpg",
        "link": "/exhibit/3"
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
    # Add more detailed exhibits as needed
]

# Flask routes
@app.route('/')
def home():
    return render_template('home.html', active_page='home')

@app.route('/exhibit/<int:exhibit_id>')
def exhibit_page(exhibit_id):
    exhibit = get_exhibit_by_id(exhibit_id)
    if exhibit:
        return render_template('exhibit.html', exhibit=exhibit, exhibit_id=exhibit_id)
    return "Exhibit not found", 404

@app.route('/tour/<int:current_id>')
def tour(current_id):
    exhibit = get_exhibit_by_id(current_id)
    if exhibit:
        next_id = current_id + 1 if current_id < len(detail_exhibits) else None
        prev_id = current_id - 1 if current_id > 1 else None
        return render_template('tour.html', exhibit=exhibit, current_id=current_id, next_id=next_id, prev_id=prev_id)
    return "Exhibit not found", 404

def get_exhibit_by_id(exhibit_id):
    for exhibit in detail_exhibits:
        if exhibit['id'] == exhibit_id:
            return exhibit
    return None

# Other routes...
@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html', active_page='chatbot')

@app.route('/get', methods=['POST'])
def query():
    user_input = request.form["msg"]

    output = query_rag_chain(user_input)

    # Return a streaming response
    return Response(stream_with_context(generate()), content_type='text/event-stream')

@app.route('/about')
def about():
    return render_template('about.html', exhibits=Exhibits, active_page='about')

@app.route('/exhibits')
def exhibits():
    return render_template('exhibits.html', exhibits=Exhibits, active_page='exhibits')

if __name__ == '__main__':
    app.run(debug=True)
