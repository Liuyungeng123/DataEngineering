import streamlit as st
import json
from cryptography.fernet import Fernet
import sqlite3
import hashlib
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
import os
import copy
import os
from datetime import datetime

import streamlit as st
from streamlit.logger import get_logger
import torch
from lagent.actions import ActionExecutor, GoogleSearch, PythonInterpreter
from lagent.agents.react import ReAct
from lagent.llms import GPTAPI
from lagent.llms.huggingface import HFTransformerCasualLM


from lagent.actions import BaseAction
# 初始化加密密钥
KEY_FILE = "encryption_key.key"
if os.path.exists(KEY_FILE):
    with open(KEY_FILE, "rb") as key_file:
        KEY = key_file.read()
else:
    KEY = Fernet.generate_key()
    with open(KEY_FILE, "wb") as key_file:
        key_file.write(KEY)

fernet = Fernet(KEY)

conn = sqlite3.connect('chat_app2s.db')
c = conn.cursor()

c.execute('''CREATE TABLE IF NOT EXISTS users
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                password TEXT)''')

c.execute('''CREATE TABLE IF NOT EXISTS conversations
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                title TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id))''')

c.execute('''CREATE TABLE IF NOT EXISTS chat_history
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                conversation_id INTEGER,
                message TEXT,
                is_user BOOLEAN,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (conversation_id) REFERENCES conversations(id))''')

c.execute('''CREATE TABLE IF NOT EXISTS conversation_settings
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER,
                system_prompt TEXT,
                temperature FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id))''')

def save_conversation_settings(conversation_id, system_prompt, temperature):
    encrypted_prompt = encrypt_data(system_prompt)
    encrypted_temp = encrypt_data(temperature)
    c.execute("""
        INSERT INTO conversation_settings (conversation_id, system_prompt, temperature)
        VALUES (?, ?, ?)""", (conversation_id, encrypted_prompt, encrypted_temp))
    conn.commit()

def get_conversation_settings(conversation_id):
    c.execute("""
        SELECT system_prompt, temperature
        FROM conversation_settings
        WHERE conversation_id = ?""", (conversation_id,))
    result = c.fetchone()
    # st.write(result, decrypt_data(result[0]), decrypt_data(result[1]))
    if result:
        return {"system_prompt": decrypt_data(result[0]), "temperature": decrypt_data(result[1]) }
    return {"system_prompt": "You are a helpful assistant.", "temperature": 0.7}
    
    
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_user(username, password):
    hashed_password = hash_password(password)
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def verify_user(username, password):
    hashed_password = hash_password(password)
    c.execute("SELECT id FROM users WHERE username = ? AND password = ?", (username, hashed_password))
    user = c.fetchone()
    return user[0] if user else None

def delete_conversation(conversation_id):
    c.execute("DELETE FROM chat_history WHERE conversation_id = ?", (conversation_id,))
    c.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    conn.commit()

def encrypt_data(data):
    if data is None:
        return None
    return fernet.encrypt(str(data).encode()).decode()

def decrypt_data(encrypted_data):
    if encrypted_data is None:
        return None
    try:
        return fernet.decrypt(encrypted_data.encode()).decode()
    except:
        return "Error: Could not decrypt data"

def save_message(user_id, message, is_user):
    encrypted_message = encrypt_message(message)
    c.execute("INSERT INTO chat_history (user_id, message, is_user) VALUES (?, ?, ?)",
              (user_id, encrypted_message, is_user))
    conn.commit()

def load_chat_history(user_id):
    c.execute("SELECT message, is_user FROM chat_history WHERE user_id = ? ORDER BY id", (user_id,))
    history = c.fetchall()
    return [(decrypt_message(msg), is_user) for msg, is_user in history]

def get_bot_response(message, history, llm):
    messages = [
        SystemMessage(content="You are a helpful assistant."),
    ]
    for msg, is_user in history:
        if is_user:
            messages.append(HumanMessage(content=msg))
        else:
            messages.append(AIMessage(content=msg))
    messages.append(HumanMessage(content=message))
    
    response = llm(messages)
    return response.content


def create_conversation(user_id, title=None):
    if title is None:
        title = "New Chat"
    encrypted_title = encrypt_data(title)
    c.execute("INSERT INTO conversations (user_id, title) VALUES (?, ?)", 
              (user_id, encrypted_title))
    conn.commit()
    return c.lastrowid

def update_conversation_title(conversation_id, title):
    encrypted_title = encrypt_data(title)
    c.execute("UPDATE conversations SET title = ? WHERE id = ?", 
              (encrypted_title, conversation_id))
    conn.commit()


def get_user_conversations(user_id):
    c.execute("""
        SELECT id, title, created_at 
        FROM conversations 
        WHERE user_id = ? 
        ORDER BY created_at DESC""", (user_id,))
    conversations = c.fetchall()
    # 解密标题
    return [(conv_id, decrypt_data(title), created_at) 
            for conv_id, title, created_at in conversations]

def is_first_message(conversation_id):
    c.execute("""SELECT COUNT(*) FROM chat_history 
                 WHERE conversation_id = ? AND is_user = 1""", (conversation_id,))
    count = c.fetchone()[0]
    return count == 0

def save_message(user_id, conversation_id, message, is_user):
    if is_user and is_first_message(conversation_id):
        title = message[:50] if len(message) > 50 else message
        update_conversation_title(conversation_id, title)
    
    encrypted_message = encrypt_data(message)
    c.execute("""
        INSERT INTO chat_history (user_id, conversation_id, message, is_user) 
        VALUES (?, ?, ?, ?)""", (user_id, conversation_id, encrypted_message, is_user))
    conn.commit()

def load_chat_history(user_id, conversation_id):
    c.execute("""
        SELECT message, is_user 
        FROM chat_history 
        WHERE user_id = ? AND conversation_id = ? 
        ORDER BY timestamp""", (user_id, conversation_id))
    history = c.fetchall()
    return [(decrypt_data(msg), is_user) for msg, is_user in history]

def login_page():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Login"):
            user_id = verify_user(username, password)
            if user_id:
                st.session_state.user_id = user_id
                st.session_state.username = username
                st.success("Successfully Login!")
                st.rerun()
            else:
                st.error("Incorrect Username or Password")
    with col2:
        if st.button("Register"):
            if create_user(username, password):
                st.success("Register successfully.  Please Login")
            else:
                st.error("Username Already Exists")


class SessionState:
    def init_state(self):
        """Initialize session state variables."""
        st.session_state["assistant"] = []
        st.session_state["user"] = []

    def clear_state(self):
        st.session_state["assistant"] = []
        st.session_state["user"] = []
        st.session_state["model_selected"] = None
        st.session_state.history = []

        if "chatbot" in st.session_state:
            st.session_state["chatbot"]._session_history = []


class StreamlitUI:

    def __init__(self, session_state: SessionState):
        self.init_streamlit()
        self.session_state = session_state

    def init_streamlit(self):
        """Initialize Streamlit's UI settings."""
        st.sidebar.title("Model Control")
    
    def setup_sidebar(self):
        """Setup the sidebar for model and plugin selection."""
        model_name = st.sidebar.selectbox(
            "Model Selection", options=["Doubao-Pro-32k", "InternLM2-20B"]
        )
        
        if st.sidebar.button("清空对话", key="clear"):
            self.session_state.clear_state()
        uploaded_file = st.sidebar.file_uploader(
            "上传文件", type=["png", "jpg", "jpeg", "mp4", "mp3", "wav"]
        )
        return model_name, uploaded_file

    def init_model(self, option):
        """Initialize the model based on the selected option."""
        if option not in st.session_state["model_map"]:
            if option.startswith("gpt"):
                st.session_state["model_map"][option] = GPTAPI(model_type=option)
            else:
                st.session_state["model_map"][option] = torch.load(
                    "your path"
                )
        return st.session_state["model_map"][option]

    def initialize_chatbot(self, model, plugin_action):
        """Initialize the chatbot with the given model and plugin actions."""
        return ReAct(llm=model, action_executor=ActionExecutor(actions=plugin_action))

    def render_user(self, prompt: str):
        with st.chat_message("user"):
            st.markdown(prompt)

    def render_assistant(self, prompt: str):
        with st.chat_message("assistant"):
            st.markdown(prompt)

    def render_action(self, action):
        with st.expander(action.type, expanded=True):
            st.markdown(
                "<p style='text-align: left;display:flex;'> <span style='font-size:14px;font-weight:600;width:70px;text-align-last: justify;'>插    件</span><span style='width:14px;text-align:left;display:block;'>:</span><span style='flex:1;'>"  # noqa E501
                + action.type
                + "</span></p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<p style='text-align: left;display:flex;'> <span style='font-size:14px;font-weight:600;width:70px;text-align-last: justify;'>思考步骤</span><span style='width:14px;text-align:left;display:block;'>:</span><span style='flex:1;'>"  # noqa E501
                + action.thought
                + "</span></p>",
                unsafe_allow_html=True,
            )
            if isinstance(action.args, dict) and "text" in action.args:
                st.markdown(
                    "<p style='text-align: left;display:flex;'><span style='font-size:14px;font-weight:600;width:70px;text-align-last: justify;'> 执行内容</span><span style='width:14px;text-align:left;display:block;'>:</span></p>",  # noqa E501
                    unsafe_allow_html=True,
                )
                st.markdown(action.args["text"])
            self.render_action_results(action)

    def render_action_results(self, action):
        """Render the results of action, including text, images, videos, and
        audios."""
        if isinstance(action.result, dict):
            st.markdown(
                "<p style='text-align: left;display:flex;'><span style='font-size:14px;font-weight:600;width:70px;text-align-last: justify;'> 执行结果</span><span style='width:14px;text-align:left;display:block;'>:</span></p>",  # noqa E501
                unsafe_allow_html=True,
            )
            if "text" in action.result:
                st.markdown(
                    "<p style='text-align: left;'>" + action.result["text"] + "</p>",
                    unsafe_allow_html=True,
                )
            if "image" in action.result:
                image_path = action.result["image"]
                image_data = open(image_path, "rb").read()
                st.image(image_data, caption="Generated Image")
            if "video" in action.result:
                video_data = action.result["video"]
                video_data = open(video_data, "rb").read()
                st.video(video_data)
            if "audio" in action.result:
                audio_data = action.result["audio"]
                audio_data = open(audio_data, "rb").read()
                st.audio(audio_data)

def chat_page():
    st.title(f"Welcome, {st.session_state.username}!")

    if "ui" not in st.session_state:
        session_state = SessionState()
        session_state.init_state()
        st.session_state["ui"] = StreamlitUI(session_state)
    
    with st.sidebar:
        if st.button("Logout"):
            del st.session_state.user_id
            del st.session_state.username
            st.rerun()
            
        st.write("Create New Chat")
        system_prompt = st.text_area("Set up AI persona (System Prompt):", 
            value="You are a helpful assistant.",
            help="Set the behavior and personality traits of AI assistants")
        temperature = st.slider("Answer randomness (Temperature):", 
            min_value=0.0, max_value=1.0, value=0.7, 
            help="A higher value of randomness in the answer will make the answer more creative, while a lower value will make the answer more certain.")
        
        if st.button("Create New Chat"):
            new_conv_id = create_conversation(st.session_state.user_id)
            save_conversation_settings(new_conv_id, system_prompt, temperature)
            st.session_state.current_conversation_id = new_conv_id
            st.rerun()
            
        conversations = get_user_conversations(st.session_state.user_id)
        if conversations:
            st.write("Select an Existing Chat:")
            for conv_id, title, created_at in conversations:
                col1, col2 = st.columns([3, 1])
                with col1:
                    if st.button(f"{title} ({created_at})", key=f"conv_{conv_id}"):
                        st.session_state.current_conversation_id = conv_id
                        st.rerun()
                with col2:
                    if st.button("Delete", key=f"del_{conv_id}"):
                        delete_conversation(conv_id)
                        if 'current_conversation_id' in st.session_state and \
                           st.session_state.current_conversation_id == conv_id:
                            del st.session_state.current_conversation_id
                        st.rerun()

    if 'current_conversation_id' not in st.session_state:
        st.info("Please create a new chat or select an existing chat")
        return

    current_settings = get_conversation_settings(st.session_state.current_conversation_id)

    model_name, uploaded_file = st.session_state["ui"].setup_sidebar()
    
    if model_name == "Doubao-Pro-32k":
        inference_server_url = "https://ark.cn-beijing.volces.com/api/v3"
        api_key = "72c991d1-8078-4c49-8b5e-7b33c4e26e04"
        model_name = "ep-20240722030254-mrqhx"
        
        llm = ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            openai_api_base=inference_server_url,
            temperature=current_settings["temperature"],
        )

    def get_bot_response(message, history, llm, system_prompt):
        messages = [
            SystemMessage(content=system_prompt),
        ]
        for msg, is_user in history:
            if is_user:
                messages.append(HumanMessage(content=msg))
            else:
                messages.append(AIMessage(content=msg))
        messages.append(HumanMessage(content=message))
        
        response = llm(messages)
        return response.content

    history = load_chat_history(st.session_state.user_id, st.session_state.current_conversation_id)
    for message, is_user in history:
        if is_user:
            st.session_state["ui"].render_user(message)
        else:
            st.session_state["ui"].render_assistant(message)

    user_input = st.chat_input("Input your message:")

    if user_input:

        save_message(st.session_state.user_id, st.session_state.current_conversation_id, 
                    user_input, True)
        st.session_state["ui"].render_user(user_input)


        bot_response = get_bot_response(user_input, history, llm, 
                                        current_settings["system_prompt"])

        save_message(st.session_state.user_id, st.session_state.current_conversation_id, 
                    bot_response, False)
        st.session_state["ui"].render_assistant(bot_response)
        st.rerun()

def main():
    if 'user_id' not in st.session_state:
        login_page()
    else:
        chat_page()

if __name__ == "__main__":
    main()