import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import pickle
from preprocessing import preprocess_text  # Importa la función de preprocesamiento

# Cargar datos
data = pd.read_csv('spam.csv')
data['processed'] = data['text'].apply(preprocess_text)

# Vectorización
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(data['processed'])
y = data['label']

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo
model = MultinomialNB()
model.fit(X_train, y_train)

# Guardar modelo y vectorizador
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

# Evaluación
predictions = model.predict(X_test)
print(classification_report(y_test, predictions))
