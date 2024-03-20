# app.py

from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')


def preprocess_text(text):
    # Tokenize the text
    tokens = word_tokenize(text.lower())
    # Remove stopwords and non-alphabetic characters
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    return tokens

# def cosine_similarity(text1, text2):


from sklearn.feature_extraction.text import TfidfVectorizer
app = Flask(__name__)

# Load the pre-trained sentence transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

@app.route('/predict', methods=['POST'])
def predict_similarity():
    # Get text inputs from the request body
    data = request.get_json()
    text1 = data['text1']
    text2 = data['text2']

    vectorizer = TfidfVectorizer(tokenizer=preprocess_text)
    tfidf_matrix = vectorizer.fit_transform([text1, text2])
    # return cosine_similarity(tfidf_matrix)[0, 1]

    # Encode the input texts
    # embeddings1 = model.encode([text1], convert_to_tensor=True)
    # embeddings2 = model.encode([text2], convert_to_tensor=True)

    # Calculate cosine similarity between the embeddings
    # similarity_score = cosine_similarity(embeddings1, embeddings2)[0][0]

    # Normalize similarity score to a scale of 0 to 1
    # normalized_similarity_score = (similarity_score + 1) / 2

    # Return the similarity score in the response
    return jsonify({"similarity score": cosine_similarity(tfidf_matrix)[0, 1]})

if __name__ == '__main__':
    app.run(debug=True)
