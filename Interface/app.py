from flask import Flask, render_template, request, Response, stream_with_context
from rag_agent import user_query

app = Flask(__name__)


# API key and file paths
data_file_path = "./Data/Database_files/Sample.pdf"
embedding_data_path = "./Data/Embedding_Store/"

detail_exhibits = [
    {
        "id": 1,
        "image": "./static/images/logo.jpg",
        "title": "Ancient Artifact",
        "short_description": "A fascinating artifact from ancient history.",
        "long_description": 
        '''
        Lorem ipsum dolor sit amet, consectetur adipiscing elit. Suspendisse hendrerit egestas velit. Etiam non consequat urna, quis consectetur justo. Curabitur condimentum nunc vel purus egestas, ac lobortis diam vehicula. Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos. Mauris pretium eleifend aliquet. Quisque sed libero erat. Duis convallis sed diam sit amet consequat. Vivamus rhoncus dui at lorem fringilla, quis viverra ligula pellentesque.

Quisque tellus erat, scelerisque ac cursus quis, tempus non dolor. Donec vel suscipit dolor. Integer egestas euismod erat euismod tempor. Morbi scelerisque quam orci, non tincidunt sem eleifend in. Aenean massa velit, tempor ut ultricies quis, volutpat quis orci. Vestibulum ultricies ante vitae diam finibus iaculis. Proin non libero quis tellus vulputate elementum. Maecenas rutrum porta velit, in pulvinar elit rutrum at. Sed tincidunt justo eros, eu mattis urna tincidunt sed. Nam vitae elit pharetra, placerat odio vel, finibus ipsum. Nunc auctor eget ligula quis sollicitudin.

Nullam rutrum et elit id luctus. Praesent porttitor laoreet lectus, non tristique velit. Nam eget libero id mi iaculis malesuada. Suspendisse vel nulla viverra, porta neque ut, eleifend dolor. Donec egestas sem non maximus feugiat. Maecenas pulvinar, turpis nec ullamcorper fermentum, turpis turpis fringilla nulla, nec bibendum orci nisi vel ligula. Sed sagittis magna eget tortor ullamcorper auctor. Nulla facilisi. Sed sit amet arcu volutpat, volutpat eros id, auctor neque. Ut magna sapien, venenatis in arcu eget, finibus dapibus neque. Donec congue erat nec mi pretium accumsan.
        '''
    },
    {
        "id": 2,
        "image": "./static/images/logo.jpg",
        "title": "Modern Art Piece",
        "short_description": "A stunning example of modern art.",
        "long_description": "This artwork represents the abstract nature of human emotions...",
        "additional_info": "Created by a famous artist in 2021, this piece is part of the XYZ collection."
    },
       {
        "id": 3,
        "image": "./static/images/logo.jpg",
        "title": "Modern Art Piece",
        "short_description": "A stunning example of modern art.",
        "long_description": "This artwork represents the abstract nature of human emotions...",
        "additional_info": "Created by a famous artist in 2021, this piece is part of the XYZ collection."
    },   {
        "id": 4,
        "image": "./static/images/logo.jpg",
        "title": "Modern Art Piece",
        "short_description": "A stunning example of modern art.",
        "long_description": "This artwork represents the abstract nature of human emotions...",
        "additional_info": "Created by a famous artist in 2021, this piece is part of the XYZ collection."
    },   {
        "id": 5,
        "image": "./static/images/logo.jpg",
        "title": "Modern Art Piece",
        "short_description": "A stunning example of modern art.",
        "long_description": "This artwork represents the abstract nature of human emotions...",
        "additional_info": "Created by a famous artist in 2021, this piece is part of the XYZ collection."
    },   {
        "id": 6,
        "image": "./static/images/logo.jpg",
        "title": "Modern Art Piece",
        "short_description": "A stunning example of modern art.",
        "long_description": "This artwork represents the abstract nature of human emotions...",
        "additional_info": "Created by a famous artist in 2021, this piece is part of the XYZ collection."
    },   {
        "id": 7,
        "image": "./static/images/logo.jpg",
        "title": "Modern Art Piece",
        "short_description": "A stunning example of modern art.",
        "long_description": "This artwork represents the abstract nature of human emotions...",
        "additional_info": "Created by a famous artist in 2021, this piece is part of the XYZ collection."
    },   {
        "id": 8,
        "image": "./static/images/logo.jpg",
        "title": "Modern Art Piece",
        "short_description": "A stunning example of modern art.",
        "long_description": "This artwork represents the abstract nature of human emotions...",
        "additional_info": "Created by a famous artist in 2021, this piece is part of the XYZ collection."
    },   {
        "id": 9,
        "image": "./static/images/logo.jpg",
        "title": "Modern Art Piece",
        "short_description": "A stunning example of modern art.",
        "long_description": "This artwork represents the abstract nature of human emotions...",
        "additional_info": "Created by a famous artist in 2021, this piece is part of the XYZ collection."
    },   {
        "id": 10,
        "image": "./static/images/logo.jpg",
        "title": "Modern Art Piece",
        "short_description": "A stunning example of modern art.",
        "long_description": "This artwork represents the abstract nature of human emotions...",
        "additional_info": "Created by a famous artist in 2021, this piece is part of the XYZ collection."
    },
    # Add more detailed exhibits as needed
]

# Flask routes
@app.route('/')
def home():
    return render_template('home.html', active_page='home')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html', active_page='chatbot')


@app.route('/about')
def about():
    return render_template('about.html', active_page='about')

@app.route('/exhibits')
def exhibits():
    return render_template('exhibits.html', exhibits=detail_exhibits, active_page='exhibits')

@app.route('/exhibit/<int:exhibit_id>')
def exhibit_page(exhibit_id):
    exhibit = get_exhibit_by_id(exhibit_id)
    if exhibit:
        return render_template('exhibit.html', exhibit=detail_exhibits, exhibit_id=exhibit_id)
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



@app.route('/get', methods=['POST'])
def query():
    user_input = request.form["msg"]

    output = user_query(user_input)
    # Return a streaming response
    # return Response(stream_with_context(generate()), content_type='text/event-stream')
    return output

if __name__ == '__main__':
    app.run(debug=True)
