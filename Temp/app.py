from flask import Flask, request, jsonify, render_template
import requests
import pdb
from table_retrieval_anirudh import table_func, knowledge_retrieval, response_generation

app = Flask(__name__)


# counter = 0
candidates = None
context = ''

def main_func(message):
    response = f"Main Function received: {message}"
    return response

def gcm(message):
    response = f"GCM Function received: {message}"
    return response

@app.route('/')
def home():
    return render_template('index_2.html')


@app.route('/process_message', methods=['POST'])
def process_message():
    global candidates
    global context 
    user_message = request.form['message']
    context = f'{context} {user_message}'
    # response = main_func(user_message)
    response, candidates = table_func(user_message)
    return jsonify(message=response)


@app.route('/api_call', methods=['POST'])
def api_call():
    global context 
    user_message = request.form['message']
    knowledge = knowledge_retrieval(user_message, candidates)
    response = response_generation(knowledge, user_message)
    context = f'{context} {user_message} {response}'
    # response = gcm(user_message)
    return jsonify(message=response)

if __name__ == '__main__':
    app.run(debug=True)



