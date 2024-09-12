from flask import Flask, render_template, request, jsonify
# from rag_chain_pipeline import query_rag_chain

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html', active_page = 'home')

@app.route('/exhibit/<exhibit_id>')
def exhibit(exhibit_id):
    # Fetch exhibit details based on exhibit_id
    return render_template('exhibit.html', exhibit_id=exhibit_id)

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html', active_page = 'chatbot')

@app.route('/about')
def about():
    return render_template('about.html', active_page = 'about')


@app.route('/get', methods=['POST'])
def query():
    input = request.form["msg"]
    # response = query_rag_chain(input)
    response = f'alnldlcldckbdbcbkdbcdbcdc  ckdcldlc'
    return jsonify({"response": response})
    
if __name__ == '__main__':
    app.run(debug=True)
