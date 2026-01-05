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
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)

download_nltk_data()

# ==============================
# LOAD MODEL & DATA
# ==============================
@st.cache_resource
def load_model():
    lemmatizer = WordNetLemmatizer()

    with open("intents.json", "r", encoding="utf-8") as f:
        intents = json.load(f)

    with open("words.pkl", "rb") as f:
        words = pickle.load(f)

    with open("classes.pkl", "rb") as f:
        classes = pickle.load(f)

    model = tf.keras.models.load_model(
        "catering_customer_service_chatbot.keras",
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
        {
            "role": "assistant",
            "content": "Halo! Selamat datang di layanan catering kami. Ada yang bisa saya bantu?"
        }
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
    res = model.predict(np.array([bow]), verbose=0)[0]
    ERROR_THRESHOLD = 0.25

    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    return [
        {"intent": classes[r[0]], "probability": str(r[1])}
        for r in results
    ]

def get_response(intents_list, intents_json, user_id="user"):
    if not intents_list:
        return "Maaf, saya belum memahami pertanyaan Anda."

    tag = intents_list[0]["intent"]

    for intent in intents_json["intents"]:
        if intent["tag"] == tag:

            # Context filter
            if "context_filter" in intent:
                if user_id not in st.session_state.context:
                    return "Maaf, saya belum memahami pertanyaan Anda."
                if intent["context_filter"] != st.session_state.context[user_id]:
                    return "Maaf, saya belum memahami pertanyaan Anda."

            # Context set
            if "context_set" in intent:
                st.session_state.context[user_id] = intent["context_set"]

            return random.choice(intent["responses"])

    return "Maaf, saya belum memahami pertanyaan Anda."

# ==============================
# UI HEADER
# ==============================
st.title("🍽️ Chatbot Catering")
st.caption("Customer Service Assistant")

# ==============================
# CHAT HISTORY (PASTI MUNCUL)
# ==============================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ==============================
# CHAT INPUT
# ==============================
if prompt := st.chat_input("Ketik pesan Anda..."):
    # tampilkan pesan user
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    # respon bot
    with st.chat_message("assistant"):
        with st.spinner("Mengetik..."):
            intents_pred = predict_class(prompt)
            reply = get_response(intents_pred, intents)
            st.markdown(reply)

    st.session_state.messages.append(
        {"role": "assistant", "content": reply}
    )

# ==============================
# SIDEBAR
# ==============================
with st.sidebar:
    st.header("ℹ️ Informasi")
    st.write("Chatbot ini dapat membantu Anda:")
    st.write("- Informasi menu catering")
    st.write("- Harga paket")
    st.write("- Cara pemesanan")
    st.write("- Jam operasional")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Halo! Selamat datang di layanan catering kami. Ada yang bisa saya bantu?"
            }
        ]
        st.session_state.context = {}
        st.rerun()
