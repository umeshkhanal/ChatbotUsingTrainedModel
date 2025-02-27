from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


app = Flask(__name__, template_folder="templates") 

# Load trained model, tokenizer & responses
model = tf.keras.models.load_model("my_model.keras")
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

# Load intents.json
import json
with open("intents.json", "r") as file:
    data = json.load(file)

responses = {intent["tag"]: intent["responses"] for intent in data["intents"]}

# Get all patterns from intents.json
all_patterns = []
pattern_to_tag = {}

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        all_patterns.append(pattern)
        pattern_to_tag[pattern] = intent["tag"]  # Store which pattern belongs to which tag

# Convert patterns into TF-IDF vectors
vectorizer = TfidfVectorizer()
vectorizer.fit(all_patterns)
pattern_vectors = vectorizer.transform(all_patterns)  # Store pattern embeddings

def find_best_matching_pattern(user_input):
    """Finds the most similar pattern to user input using cosine similarity."""
    user_vector = vectorizer.transform([user_input])  # Convert input to TF-IDF vector
    similarities = cosine_similarity(user_vector, pattern_vectors)  # Compute similarity
    best_match_index = np.argmax(similarities)  # Find highest similarity score
    return all_patterns[best_match_index]  # Return the closest matched pattern

def predict_intent(user_input):
    """Predicts intent using the best matched pattern."""
    best_match = find_best_matching_pattern(user_input)  # Find closest match
    predicted_tag = pattern_to_tag[best_match]  # Get its intent tag
    return predicted_tag

def get_response(intent_tag):
    """Returns all responses for a given intent."""
    return responses.get(intent_tag, ["I'm sorry, I don't understand."])

@app.route("/")
def home():
    return render_template("index.html")  # Redirect to index.html

@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json()
    user_input = data.get("message", "")
    
    predicted_tag = predict_intent(user_input)
    response_list = get_response(predicted_tag)

    return jsonify({"response": response_list})
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
