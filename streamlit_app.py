import streamlit as st
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import tensorflow as tf

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)

download_nltk_data()

# Load model and data (cache untuk performance)
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
        "catering_customer_service_chatbot.keras",
        compile=False
    )
    
    return lemmatizer, intents, words, classes, model

lemmatizer, intents, words, classes, model = load_model()

# Context dictionary
if 'context' not in st.session_state:
    st.session_state.context = {}

# Chat history
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Halo! Selamat datang di layanan catering kami. Ada yang bisa saya bantu?"}
    ]

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
    
    tag = intents_list[0]['intent']
    
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            if "context_filter" in intent:
                if user_id not in st.session_state.context:
                    continue
                if intent["context_filter"] != st.session_state.context[user_id]:
                    continue
            
            if "context_set" in intent:
                st.session_state.context[user_id] = intent["context_set"]
            
            return random.choice(intent['responses'])
    
    return "Maaf, saya belum memahami pertanyaan Anda."

# Page config
st.set_page_config(
    page_title="Chatbot Catering Service",
    page_icon="🍽️",
    layout="centered"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #2b313e;
        align-items: flex-end;
    }
    .chat-message.bot {
        background-color: #475063;
        align-items: flex-start;
    }
    .chat-message .message {
        max-width: 80%;
        padding: 0.5rem 1rem;
        border-radius: 1rem;
    }
    .chat-message.user .message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .chat-message.bot .message {
        background-color: white;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# Header
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.title("🍽️ Chatbot Catering")
    st.caption("Customer Service Assistant")

# Display chat messages
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user">
            <div class="message">{content}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message bot">
            <div class="message">{content}</div>
        </div>
        """, unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ketik pesan Anda..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get bot response
    with st.spinner("Mengetik..."):
        ints = predict_class(prompt)
        response = get_response(ints, intents)
    
    # Add bot response
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Rerun to update chat
    st.rerun()

# Sidebar
with st.sidebar:
    st.header("ℹ️ Informasi")
    st.write("Chatbot ini dapat membantu Anda dengan:")
    st.write("- Informasi menu catering")
    st.write("- Harga paket")
    st.write("- Cara pemesanan")
    st.write("- Jam operasional")
    st.write("- Dan lainnya!")
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Halo! Selamat datang di layanan catering kami. Ada yang bisa saya bantu?"}
        ]
        st.session_state.context = {}
        st.rerun()
