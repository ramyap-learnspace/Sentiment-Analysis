import os
import re
import nltk
import spacy
import pickle as pk
import pandas as pd
import requests

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load spaCy model
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])

# Create pickle folder if not exists
os.makedirs('pickle_file', exist_ok=True)

# Only download `model.pkl` if missing
MODEL_URL = "https://drive.google.com/file/d/1LvmP5v0OiPxHnbj27azpMumYK_bFs0hC/view?usp=sharing"  
model_path = os.path.join("pickle_file", "model.pkl")
if not os.path.exists(model_path):
    print("Downloading model.pkl...")
    r = requests.get(MODEL_URL)
    if r.status_code == 200:
        with open(model_path, "wb") as f:
            f.write(r.content)
    else:
        raise Exception(f"Failed to download model.pkl: {r.status_code}")

# Load pickle files
count_vector = pk.load(open("pickle_file/count_vector.pkl", "rb"))
tfidf_transformer = pk.load(open("pickle_file/tfidf_transformer.pkl", "rb"))
model = pk.load(open("pickle_file/model.pkl", "rb"))
recommend_matrix = pk.load(open("pickle_file/user_final_rating.pkl", "rb"))

# Load product data
product_df = pd.read_csv("sample30.csv")

# Cleaning functions
def remove_special_characters(text, remove_digits=True):
    pattern = r'[^a-zA-Z\s]' if remove_digits else r'[^a-zA-Z0-9\s]'
    return re.sub(pattern, '', text)

def to_lowercase(words):
    return [word.lower() for word in words]

def remove_punctuation_and_splchars(words):
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word:
            new_word = remove_special_characters(new_word, True)
            new_words.append(new_word)
    return new_words

stopword_list = stopwords.words('english')

def remove_stopwords(words):
    return [word for word in words if word not in stopword_list]

def stem_words(words):
    return [LancasterStemmer().stem(word) for word in words]

def lemmatize_verbs(words):
    return [WordNetLemmatizer().lemmatize(word, pos='v') for word in words]

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

def model_predict(text_series):
    word_vector = count_vector.transform(text_series)
    tfidf_vector = tfidf_transformer.transform(word_vector)
    return model.predict(tfidf_vector)

def recommend_products(user_name):
    product_list = pd.DataFrame(recommend_matrix.loc[user_name].sort_values(ascending=False)[0:20])
    product_frame = product_df[product_df.name.isin(product_list.index.tolist())]
    output_df = product_frame[['name', 'reviews_text']]
    output_df['lemmatized_text'] = output_df['reviews_text'].map(normalize_and_lemmaize)
    output_df['predicted_sentiment'] = model_predict(output_df['lemmatized_text'])
    return output_df

def top5_products(df):
    total_product = df.groupby(['name']).agg('count')
    rec_df = df.groupby(['name', 'predicted_sentiment']).agg('count').reset_index()
    merge_df = pd.merge(rec_df, total_product['reviews_text'], on='name')
    merge_df['%percentage'] = (merge_df['reviews_text_x'] / merge_df['reviews_text_y']) * 100
    merge_df = merge_df.sort_values(ascending=False, by='%percentage')
    output_products = pd.DataFrame(merge_df['name'][merge_df['predicted_sentiment'] == 1][:5])
    return output_products
