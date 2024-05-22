import streamlit as st
import pickle
from preprocessing import preprocess_text  # Asegúrate de que este import funcione correctamente

# Cargar modelo y vectorizador
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

def predict_spam(message):
    processed_message = preprocess_text(message)
    vectorized_message = tfidf.transform([processed_message])
    prediction = model.predict(vectorized_message)
    return "SPAM" if prediction[0] == 1 else "NO SPAM"

# Interfaz de Streamlit
st.title("Clasificacion SPAM SMS/EMAIL")
input_message = st.text_area("Ingresa tu mensaje")
if st.button('Predecir'):
    result = predict_spam(input_message)
    st.header(result)

