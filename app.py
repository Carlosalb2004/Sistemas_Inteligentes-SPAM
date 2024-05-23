import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import numpy as np

# Preprocesamiento de texto
def preprocess_text(text):
    return text.lower()

# Cargar datos y preparar modelos
@st.cache(allow_output_mutation=True)
def load_data_and_models():
    data = pd.read_csv('spam.csv', encoding='latin1')
    data = data[['v1', 'v2']]
    data.columns = ['label', 'text']
    data['text'] = data['text'].apply(preprocess_text)
    
    X = data['text']
    y = data['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'Logistic Regression': LogisticRegression(),
        'Neural Network': MLPClassifier(max_iter=1000),
        'SVM': SVC(kernel='linear')
    }
    
    trained_models = {}
    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train)
    
    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        trained_models[name] = model
    
    return tfidf, trained_models

tfidf, models = load_data_and_models()

# Streamlit UI
st.title('Email Spam Detection')
message = st.text_area("Enter the message you want to classify:")

if st.button("Classify"):
    message_transformed = tfidf.transform([preprocess_text(message)])
    
    # Predictions from all models
    predictions = [model.predict(message_transformed)[0] for model in models.values()]
    
    # Majority vote
    prediction = max(set(predictions), key=predictions.count)
    
    st.write(f"The message is predicted as: {prediction}")
