import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def preprocess_text(text):
    text = text.lower()
    words = word_tokenize(text)
    filtered_words = [ps.stem(word) for word in words if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_words)
