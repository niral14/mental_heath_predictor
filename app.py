import streamlit as st
import pickle
import numpy as np
from scipy.sparse import hstack

# Load saved vectorizer and model
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Class labels in the correct order
class_labels = [
    "Anxiety",
    "Bipolar",
    "Depression",
    "Normal",
    "Personality disorder",
    "Stress",
    "Suicidal"
]

# Preprocessing function
def preprocess_text(text):
    return text  # You can add tokenization/stemming here if needed

st.title("🧠 Mental Health Statement Classifier")

input_text = st.text_area("Enter a statement related to mental health:")

if st.button("Predict"):
    if input_text.strip() == "":
        st.error("❗ Please enter a valid statement!")
    else:
        # Step 1: Preprocess input
        processed_text = preprocess_text(input_text)

        # Step 2: Vectorize the input
        input_tfidf = vectorizer.transform([processed_text])

        # Step 3: Extract numerical features
        num_of_characters = len(input_text)
        num_of_sentences = input_text.count('.') + input_text.count('!') + input_text.count('?')

        input_num = np.array([[num_of_characters, num_of_sentences]])

        # Step 4: Combine features
        input_vector = hstack([input_tfidf, input_num])

        # Step 5: Predict
        predicted_class_index = model.predict(input_vector)[0]
        predicted_class_label = class_labels[predicted_class_index]

        st.success(f"✅ Prediction: {predicted_class_label}")
