import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def preprocess_text(text):
    # Aquí puedes agregar la limpieza del texto según tu necesidad
    # Por ahora, solo convertimos el texto a minúsculas
    return text.lower()

# Cargar los datos
df = pd.read_csv('spam.csv', encoding='latin-1')
df.dropna(axis=1, inplace=True)  # Asegúrate de que el DataFrame no tenga columnas extras vacías

# Preprocesar los textos
df['processed_text'] = df['v2'].apply(preprocess_text)  # Asumo que 'v2' es la columna con los textos

# Inicializar y entrenar TfidfVectorizer
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['processed_text'])

# Guardar el vectorizador entrenado
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

print("Vectorizer has been trained and saved successfully.")
