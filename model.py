# model.py

# Importing Libraries
import pandas as pd
import re
import nltk
import spacy
import string
import os
import pickle as pk
import subprocess
import requests

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer
from nltk.stem import WordNetLemmatizer

# Setup NLTK data directory
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Download NLTK resources if not already present
for resource in ['punkt', 'stopwords', 'wordnet', 'omw-1.4']:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource, download_dir=nltk_data_dir)

# Download spaCy model if missing
try:
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
except:
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])

# Load CSV
product_df = pd.read_csv('sample30.csv', sep=",")

# --- Preprocessing Utilities ---
def remove_special_characters(text, remove_digits=True):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text

def to_lowercase(words):
    return [word.lower() for word in words]

def remove_punctuation_and_splchars(words):
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_word = remove_special_characters(new_word, True)
            new_words.append(new_word)
    return new_words

stopword_list = stopwords.words('english')

def remove_stopwords(words):
    return [word for word in words if word not in stopword_list]

def stem_words(words):
    stemmer = LancasterStemmer()
    return [stemmer.stem(word) for word in words]

def lemmatize_verbs(words):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word, pos='v') for word in words]

def normalize(words):
    words = to_lowercase(words)
    words = remove_punctuation_and_splchars(words)
    words = remove_stopwords(words)
    return words

def lemmatize(words):
    return lemmatize_verbs(words)

def normalize_and_lemmaize(input_text):
    input_text = remove_special_characters(input_text)
    words = nltk.word_tokenize(input_text)
    words = normalize(words)
    lemmas = lemmatize(words)
    return ' '.join(lemmas)

# --- Load Models ---

# Load CountVectorizer & TFIDF
count_vector = pk.load(open('pickle_file/count_vector.pkl', 'rb'))
tfidf_transformer = pk.load(open('pickle_file/tfidf_transformer.pkl', 'rb'))

# Download and cache model.pkl from Google Drive (if not present)
model_path = 'pickle_file/model.pkl'
if not os.path.exists(model_path):
    print("Downloading model.pkl from Google Drive...")
    model_url = 'https://drive.google.com/uc?id=1LvmP5v0OiPxHnbj27azpMumYK_bFs0hC'
    r = requests.get(model_url, allow_redirects=True)
    with open(model_path, 'wb') as f:
        f.write(r.content)

# Load classification model
model = pk.load(open(model_path, 'rb'))

# --- Prediction & Recommendation Functions ---
def model_predict(text):
    word_vector = count_vector.transform(text)
    tfidf_vector = tfidf_transformer.transform(word_vector)
    output = model.predict(tfidf_vector)
    return output

def recommend_products(user_name):
    recommend_matrix = pk.load(open('pickle_file/user_final_rating.pkl', 'rb'))
    product_list = pd.DataFrame(recommend_matrix.loc[user_name].sort_values(ascending=False)[0:20])
    product_frame = product_df[product_df.name.isin(product_list.index.tolist())]
    output_df = product_frame[['name', 'reviews_text']].copy()
    output_df['lemmatized_text'] = output_df['reviews_text'].map(lambda text: normalize_and_lemmaize(text))
    output_df['predicted_sentiment'] = model_predict(output_df['lemmatized_text'])
    return output_df

def top5_products(df):
    total_product = df.groupby(['name']).agg('count')
    rec_df = df.groupby(['name', 'predicted_sentiment']).agg('count')
    rec_df = rec_df.reset_index()
    merge_df = pd.merge(rec_df, total_product['reviews_text'], on='name')
    merge_df['%percentage'] = (merge_df['reviews_text_x'] / merge_df['reviews_text_y']) * 100
    merge_df = merge_df.sort_values(ascending=False, by='%percentage')
    output_products = pd.DataFrame(merge_df['name'][merge_df['predicted_sentiment'] == 1][:5])
    return output_products
