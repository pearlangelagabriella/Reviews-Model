import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer 
from nltk.stem.wordnet import WordNetLemmatizer

# Load the model and vectorizer
with open('../Output/model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('../Output/vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

def preprocess_text(input_text, use_stemming=True, use_lemmatization=True):
    if not isinstance(input_text, str):
        input_text = str(input_text)

    # Convert to lowercase
    preprocessed_text = input_text.lower()

    # Remove punctuation
    preprocessed_text = re.sub(r'[^\w\s]', '', preprocessed_text)
                               
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    stop_words.remove("not")
    word_tokens = word_tokenize(preprocessed_text)
    filtered_words = [word for word in word_tokens if word not in stop_words]

    # Initialize stemmer and lemmatizer
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    
    # Optionally apply stemming and lemmatization
    if use_stemming:
        filtered_words = [stemmer.stem(word) for word in filtered_words]
    if use_lemmatization:
        filtered_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    
    preprocessed_text = ' '.join(filtered_words)
    
    return preprocessed_text

def main():
    st.title("Skincare Product Review Rating Predictor")

    st.markdown("""
## Skincare Product Review Rating Predictor
This app predicts the rating of skincare product reviews based on the text you provide. 
Please enter your review in the text area below and click the "Predict" button to get the predicted rating.
""")

    review = st.text_area("Enter your review:")
    if st.button("Predict"):
        if review:
            # Preprocess the input
            processed_review = preprocess_text(review)

            # Vectorize the input
            review_vector = vectorizer.transform([processed_review])  # Wrap it in a list

            # Make prediction
            prediction = model.predict(review_vector)[0]  # This line is correct
           
            try:
                # Make prediction
                prediction = model.predict(review_vector)[0]
                st.write(f"Predicted Rating: {prediction}")
            except Exception as e:
                st.write(f"Error during prediction: {e}")
        else:
            st.write("Please enter a review.")


if __name__ == "__main__":
    main()
