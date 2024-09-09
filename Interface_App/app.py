from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/exhibit/<exhibit_id>')
def exhibit(exhibit_id):
    # Fetch exhibit details based on exhibit_id
    return render_template('exhibit.html', exhibit_id=exhibit_id)

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

if __name__ == '__main__':
    app.run(debug=True)
