import streamlit as st
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
import streamlit.components.v1 as components

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(
    page_title="Chatbot Catering",
    page_icon="🍽️",
    layout="centered"
)

# ==============================
# NLTK
# ==============================
@st.cache_resource
def download_nltk():
    for r in ["punkt", "punkt_tab", "wordnet"]:
        try:
            nltk.data.find(r)
        except:
            nltk.download(r, quiet=True)

download_nltk()

# ==============================
# LOAD MODEL
# ==============================
@st.cache_resource
def load_model():
    lemmatizer = WordNetLemmatizer()
    intents = json.load(open("intents.json", encoding="utf-8"))
    words = pickle.load(open("words.pkl", "rb"))
    classes = pickle.load(open("classes.pkl", "rb"))
    model = tf.keras.models.load_model(
        "catering_customer_service_chatbot.keras",
        compile=False
    )
    return lemmatizer, intents, words, classes, model

lemmatizer, intents, words, classes, model = load_model()

# ==============================
# SESSION STATE
# ==============================
if "messages" not in st.session_state:
    st.session_state.messages = [
        ("bot", "Halo 👋 Saya asisten layanan pelanggan catering.")
    ]

# ==============================
# NLP
# ==============================
def clean(sentence):
    return [lemmatizer.lemmatize(w.lower()) for w in nltk.word_tokenize(sentence)]

def bow(sentence):
    s_words = clean(sentence)
    bag = [0]*len(words)
    for s in s_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict(sentence):
    res = model.predict(np.array([bow(sentence)]), verbose=0)[0]
    idx = np.argmax(res)
    return classes[idx]

def respond(msg):
    tag = predict(msg)
    for i in intents["intents"]:
        if i["tag"] == tag:
            return random.choice(i["responses"])
    return "Maaf, saya belum memahami pertanyaan Anda."

# ==============================
# HTML UI
# ==============================
chat_html = """
<style>
.chatbox{
    max-width:600px;
    margin:auto;
    background:#fdfdfd;
    border-radius:20px;
    padding:20px;
}
.bubble{
    padding:10px 15px;
    border-radius:15px;
    margin:5px 0;
    max-width:80%;
}
.user{background:#dcf8c6;margin-left:auto;}
.bot{background:#eee;}
</style>
<div class="chatbox">
"""
for role,msg in st.session_state.messages:
    chat_html += f'<div class="bubble {role}">{msg}</div>'
chat_html += "</div>"

components.html(chat_html, height=400)

# ==============================
# INPUT
# ==============================
user_msg = st.text_input("Ketik pesan Anda")

if st.button("Kirim") and user_msg:
    st.session_state.messages.append(("user", user_msg))
    bot_reply = respond(user_msg)
    st.session_state.messages.append(("bot", bot_reply))
    st.rerun()
