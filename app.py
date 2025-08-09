import streamlit as st
import pickle
import time
import re
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


# --- Model and Vectorizer Loading ---
# This function loads your pre-trained TF-IDF vectorizer and model.
# st.cache_resource is used to load these resources only once.

@st.cache_resource
def load_artifacts():
    """Loads the pre-trained model and vectorizer from your pickle files."""
    try:
        # Update filenames to match your second notebook
        with open('TfIdf.pickle', 'rb') as f:
            vectorizer = pickle.load(f)
        with open('model.pickle', 'rb') as f:
            model = pickle.load(f)
        return vectorizer, model
    except FileNotFoundError:
        st.error("Model or vectorizer files not found. Please make sure 'TfIdf.pickle' and 'model.pickle' are in the same directory as the script.")
        return None, None

# Load the artifacts. The app will stop if files are not found.
vectorizer, model = load_artifacts()

# --- Text Preprocessing Function ---
# This function should be identical to the one used during training.
lemmatizer = WordNetLemmatizer()
def preprocess_text(text):
    """Cleans and preprocesses the input text."""
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = [lemmatizer.lemmatize(word.lower(), pos='v') for word in review.split() if word.lower() not in nltk.corpus.stopwords.words("english")]
    return " ".join(review)


# --- Streamlit App Interface ---

def main():
    """Defines the Streamlit application interface."""
    
    # Page configuration
    st.set_page_config(page_title="Spam Classifier", page_icon="📧", layout="centered")

    # --- Header ---
    st.title("📧 Spam or Ham Classifier")
    st.markdown("""
    This app uses your pre-trained **TF-IDF** model and **Multinomial Naive Bayes** classifier to predict whether a message is spam. 
    Enter a message below to see the prediction.
    """)
    st.write("") # Add some space

    # --- User Input ---
    message_input = st.text_area(
        "Enter your message here:", 
        "URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot! Txt the word: CLAIM to No: 81010",
        height=150
    )

    # --- Prediction Button and Output ---
    if st.button("Classify Message", use_container_width=True, type="primary"):
        if message_input and vectorizer and model:
            with st.spinner('Analyzing your message...'):
                time.sleep(1) # Simulate processing time

                # 1. Preprocess the user's input message
                processed_message = preprocess_text(message_input)

                # 2. Vectorize the processed message using the TF-IDF vectorizer
                message_vectorized = vectorizer.transform([processed_message])

                # 3. Predict the class (True for spam, False for ham)
                prediction = model.predict(message_vectorized)[0]
                
                # 4. Get the prediction probabilities
                prediction_proba = model.predict_proba(message_vectorized)

                # 5. Display the result
                st.subheader("Result")
                # Updated logic to handle boolean prediction
                if prediction: # This will be True for spam
                    # The second element in prediction_proba corresponds to the 'True' (spam) class
                    spam_probability = prediction_proba[0][1]
                    st.error(f"This looks like Spam.", icon="🚨")
                    st.progress(spam_probability)
                    st.write(f"**Confidence:** {spam_probability*100:.2f}%")
                else: # This will be False for ham
                    # The first element in prediction_proba corresponds to the 'False' (ham) class
                    ham_probability = prediction_proba[0][0]
                    st.success(f"This looks like Ham (Not Spam).", icon="✅")
                    st.progress(ham_probability)
                    st.write(f"**Confidence:** {ham_probability*100:.2f}%")
        elif not message_input:
            st.warning("Please enter a message to classify.", icon="⚠️")

    # --- Footer ---
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit and Scikit-learn.")


if __name__ == '__main__':
    # Only run the main app if the model and vectorizer were loaded successfully
    if vectorizer and model:
        main()
