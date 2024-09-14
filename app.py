from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

app = Flask(__name__)

# Load the pre-trained model
model = load_model('sentiment_model.h5')

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Define parameters
vocab_size = 10000
maxlen = 100

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        review = request.form['text']
        # Tokenize and pad the review
        sequence = tokenizer.texts_to_sequences([review])
        padded_sequence = pad_sequences(sequence, maxlen=maxlen)
        # Predict sentiment
        sentiment = model.predict(padded_sequence)
        sentiment_label = "Positive" if sentiment <= 0.5 else "Negative"
        return render_template('index.html', prediction_text=f'Sentiment: {sentiment_label}')

if __name__ == '__main__':
    app.run(debug=True)
