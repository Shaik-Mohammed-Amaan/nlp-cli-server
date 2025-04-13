from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# Initialize Flask app
app = Flask(__name__)

# Load models
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
qa_pipeline = pipeline('question-answering', model='deepset/bert-base-cased-squad2')

# API endpoint to process questions and Ollama responses
@app.route('/process', methods=['POST'])
def process():
    data = request.json
    question = data.get('question')  # User input
    ollama_response = data.get('answer')  # Ollama response

    # Use the Ollama response as the context for question-answering
    output = qa_pipeline({
        'question': question,
        'context': ollama_response
    })

    return jsonify(output)

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)