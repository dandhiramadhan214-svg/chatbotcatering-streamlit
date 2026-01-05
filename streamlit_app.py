import streamlit as st
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf

# ==============================
# PAGE CONFIG (HARUS PALING ATAS)
# ==============================
st.set_page_config(
    page_title="Chatbot Catering Service",
    page_icon="🍽️",
    layout="centered"
)

# ==============================
# NLTK SETUP
# ==============================
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)

download_nltk_data()

# ==============================
# LOAD MODEL & DATA
# ==============================
@st.cache_resource
def load_model():
    lemmatizer = WordNetLemmatizer()

    with open('intents.json', 'r', encoding='utf-8') as f:
        intents = json.load(f)

    with open('words.pkl', 'rb') as f:
        words = pickle.load(f)

    with open('classes.pkl', 'rb') as f:
        classes = pickle.load(f)

    model = tf.keras.models.load_model(
        "catering_customer_service_chatbot.keras",  # ✅ NAMA FIX
        compile=False
    )

    return lemmatizer, intents, words, classes, model

lemmatizer, intents, words, classes, model = load_model()

# ==============================
# SESSION STATE
# ==============================
if "context" not in st.session_state:
    st.session_state.context = {}

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Halo! Selamat datang di layanan catering kami. Ada yang bisa saya bantu?"}
    ]

# ==============================
# NLP FUNCTIONS
# ==============================
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]), verbose=0)
