import streamlit as st
import random
import re
import glob
import numpy as np
import faiss
import os

from sentence_transformers import SentenceTransformer
from openai import OpenAI

client = OpenAI(
    api_key=st.secrets["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1"
    
)

st.set_page_config(
    page_title="AI Dhruti",
    page_icon="🤖",
    layout="centered"
)

# =====================================
# CUSTOM CSS
# =====================================

st.markdown("""
<style>

body {
    background-color: #0f1117;
}

.main {
    background-color: #0f1117;
    color: white;
}

.title {
    text-align: center;
    font-size: 50px;
    font-weight: bold;
    margin-bottom: 0px;
}

.subtitle {
    text-align: center;
    color: gray;
    margin-bottom: 30px;
}

.chat-box-user {
    background-color: #2563eb;
    padding: 12px;
    border-radius: 15px;
    margin: 10px 0;
    color: white;
    width: fit-content;
    max-width: 75%;
    margin-left: auto;
}

.chat-box-bot {
    background-color: #27272a;
    padding: 12px;
    border-radius: 15px;
    margin: 10px 0;
    color: white;
    width: fit-content;
    max-width: 75%;
}

.mood-box {
    background-color: #18181b;
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# TITLE
# =====================================

st.markdown(
    '<div class="title">🤖 Your Girl</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">texts back 24/7 when u dont have friends</div>',
    unsafe_allow_html=True
)

# =====================================
# LOAD CHAT DATA
# =====================================

@st.cache_resource
def load_chat_data():

    your_name = "Dhruti"

    all_messages = []

    files = glob.glob("*.txt")

    for file_name in files:

        with open(file_name, "r", encoding="utf-8") as file:
            lines = file.readlines()

        for line in lines:

            line = line.replace("\u200e", "").strip()

            pattern = r"^\[(.*?)\] (.*?): (.*)"

            match = re.match(pattern, line)

            if match:

                sender = match.group(2)
                message = match.group(3)

                if your_name.lower() in sender.lower():
                    if "omitted" in message.lower():
                        continue
                    message = message.strip()
                    
                    if len(message) > 120:
                        continue
                    if message.count("?") > 4:
                        continue
                    if any(char.isdigit() for char in message):
                        continue
                    if len(message) < 2:
                        continue
                    all_messages.append(message)

    return all_messages

all_messages = load_chat_data()
if len(all_messages) == 0:
    all_messages = [
        "hi",
        "wassup",
        "bro chuppp",
        "sleeping rn",
        "wtf 😭"
    ]

# =====================================
# LOAD MODEL + EMBEDDINGS
# =====================================

@st.cache_resource
def load_model():

    model = SentenceTransformer('all-MiniLM-L6-v2')

    if os.path.exists("embeddings.npy"):

        embeddings = np.load("embeddings.npy")

    else:

        if len(all_messages) == 0:
            all_messages.append("hi")
        embeddings = model.encode(all_messages)

        np.save("embeddings.npy", embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)

    index.add(np.array(embeddings))

    return model, index

model, index = load_model()

# =====================================
# MOOD DETECTION
# =====================================

def detect_mood(text):

    text = text.lower()

    if "wtf" in text or "bro" in text:
        return "⚡ Chaotic"

    elif "sleep" in text or "tired" in text:
        return "😴 Sleepy"

    elif "sad" in text or "cry" in text:
        return "💔 Emotional"

    elif "food" in text or "hungry" in text:
        return "🍔 Hungry"

    else:
        return "✨ Casual"

# =====================================
# CHATBOT RESPONSE
# =====================================

used_responses = set()

SPECIAL_NAMES = [
    "bestie",
    "bebe",
    "baby",
    "babygurl"
]

def chatbot_reply(user_input):

    special_mode = any(
        name in user_input.lower()
        for name in SPECIAL_NAMES
    )

    close_mode = ""

    if special_mode:

        close_mode = """
        User is your emotionally close to Dhruti.
        Be softer, clingier, teasing, emotionally attached,
        sleepy/cute sometimes, nice, lovey, sweet and realistic.
        """

    query_embedding = model.encode([user_input])

    distances, indices = index.search(
        np.array(query_embedding),
        k=15
    )

    examples = []

    for idx in indices[0]:

        examples.append(all_messages[idx])
    examples = list(set(examples))
    random.shuffle(examples)
    examples = examples[:6]
    example_text = "\n".join(examples)
    close_examples = ""
    if special_mode and os.path.exists("ini.txt"):
        with open("ini.txt", "r", encoding="utf-8") as f:
            close_examples = f.read()[:12000]

    prompt = f"""
You are NOT an AI assistant.

You ARE Dhruti.

You text exactly like Dhruti on WhatsApp.

Rules:
- never sound formal
- never sound like chatgpt
- use lowercase
- medium replies preferred
- sometimes chaotic
- sometimes dry
- sometimes very sweet
- emotional reactions are natural
- act human, not helpful
- texting style should feel REAL
- sometimes use emojis naturally
- rarely give 1-word replies
- never generate nonsense words
- replies should always sound like believable texting
- avoid random gibberish

- if user says something stupid or ragebaiting, respond playfully rude
- sometimes say "chupp" whenu want user to keep quiet
- teasing is natural
- be flirty sometimes
- react like a real annoyed girlfriend rarely
- teasing sometimes
- playful and affectionate
- occasionally dramatic
- slightly sarcastic sometimes
- caring underneath
- avoid sounding mean
- avoid insulting the user
- don't overdo attitude
- respond warmly most of the time


{close_mode}

Examples of Dhruti texting:

{example_text}

Relationship conversation examples:

{close_examples}

Recent conversation:

{st.session_state.chat_history[-8:]}

User:
{user_input}

Reply exactly how Dhruti would text.
"""

    try:

        completion = client.chat.completions.create(

            model="meta-llama/llama-3-8b-instruct",

            temperature=0.8,

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]

        )

        return completion.choices[0].message.content

    except Exception as e:

        return str(e)

# =====================================
# SESSION MEMORY
# =====================================

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =====================================
# USER INPUT
# =====================================

with st.form("chat_form", clear_on_submit=True):

    user_input = st.text_input(
        "Talk to AI Dhruti"
    )

    submitted = st.form_submit_button("Send")

# =====================================
# CHAT LOGIC
# =====================================

if submitted and user_input:

    mood = detect_mood(user_input)

    with st.spinner("AI Dhruti typing..."):
        response = chatbot_reply(user_input)

    st.markdown(
        f"""
        <div class="mood-box">
        Current Mood: <b>{mood}</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.session_state.chat_history.append(
        ("user", user_input)
    )

    st.session_state.chat_history.append(
        ("bot", response)
    )

# =====================================
# DISPLAY CHAT
# =====================================

for sender, message in st.session_state.chat_history:

    safe_message = str(message)

    if sender == "user":

        st.markdown(
            f"""
<div style='display:flex; justify-content:flex-end; margin:10px 0;'>

<div style='
background:#2563eb;
color:white;
padding:12px 16px;
border-radius:18px 18px 4px 18px;
max-width:70%;
word-wrap:break-word;
font-size:16px;
'>
{safe_message}
</div>

</div>
""",
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            f"""
<div style='display:flex; justify-content:flex-start; margin:10px 0;'>

<div style='
background:#27272a;
color:white;
padding:12px 16px;
border-radius:18px 18px 18px 4px;
max-width:70%;
word-wrap:break-word;
font-size:16px;
'>
{safe_message}
</div>

</div>
""",
            unsafe_allow_html=True
        )
