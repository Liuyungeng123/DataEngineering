import streamlit as st
import json
from cryptography.fernet import Fernet
import sqlite3
import hashlib
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
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

from nltk.sentiment import SentimentIntensityAnalyzer

import requests
from bs4 import BeautifulSoup

sentiment_analyzer = SentimentIntensityAnalyzer()

import requests


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

# DB preparation
conn = sqlite3.connect("chat_app2s.1.3.db")
c = conn.cursor()

c.execute(
    """CREATE TABLE IF NOT EXISTS users
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE,
                password TEXT)"""
)

c.execute(
    """CREATE TABLE IF NOT EXISTS user_preference
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                preference TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id))"""
)

c.execute(
    """CREATE TABLE IF NOT EXISTS conversations
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                title TEXT,
                use_user_preferences BOOLEAN,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id))"""
)

c.execute(
    """CREATE TABLE IF NOT EXISTS chat_history
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                conversation_id INTEGER,
                message TEXT,
                is_user BOOLEAN,
                sentiment_score INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (conversation_id) REFERENCES conversations(id))"""
)

c.execute(
    """CREATE TABLE IF NOT EXISTS conversation_settings
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER,
                system_prompt TEXT,
                temperature FLOAT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id))"""
)


def save_conversation_settings(conversation_id, system_prompt, temperature):
    encrypted_prompt = encrypt_data(system_prompt)
    encrypted_temp = encrypt_data(temperature)
    c.execute(
        """
        INSERT INTO conversation_settings (conversation_id, system_prompt, temperature)
        VALUES (?, ?, ?)""",
        (conversation_id, encrypted_prompt, encrypted_temp),
    )
    conn.commit()


def get_conversation_settings(conversation_id):
    c.execute(
        """
        SELECT system_prompt, temperature
        FROM conversation_settings
        WHERE conversation_id = ?""",
        (conversation_id,),
    )
    result = c.fetchone()
    # st.write(result, decrypt_data(result[0]), decrypt_data(result[1]))
    if result:
        return {
            "system_prompt": decrypt_data(result[0]),
            "temperature": decrypt_data(result[1]),
        }
    return {"system_prompt": "You are a helpful assistant.", "temperature": 0.7}


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def create_user(username, password, others_preference):
    hashed_password = hash_password(password)
    try:
        c.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, hashed_password),
        )

        if len(others_preference) > 0:
            user = c.lastrowid
            print(others_preference)
            for preference in others_preference:
                encrypt_preference = encrypt_data(preference)
                c.execute(
                    "INSERT INTO user_preference (user_id, preference) VALUES (?, ?)",
                    (user, encrypt_preference),
                )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False


def get_user_preference(user_id):
    c.execute(
        "SELECT user_id, preference FROM user_preference WHERE user_id = ?",
        (user_id,),
    )
    user_preference = c.fetchall()
    return [
        (user_id, decrypt_data(preference)) for user_id, preference in user_preference
    ]


def verify_user(username, password):
    hashed_password = hash_password(password)
    c.execute(
        "SELECT id FROM users WHERE username = ? AND password = ?",
        (username, hashed_password),
    )
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


# seem useless
# def save_message(user_id, message, is_user):
#    encrypted_message = encrypt_message(message)
#    c.execute("INSERT INTO chat_history (user_id, message, is_user) VALUES (?, ?, ?)",
#              (user_id, encrypted_message, is_user))
#    conn.commit()


# def load_chat_history(user_id):
#    c.execute(
#        "SELECT message, is_user FROM chat_history WHERE user_id = ? ORDER BY id",
#        (user_id,),
#    )
#    history = c.fetchall()
#    return [(decrypt_message(msg), is_user) for msg, is_user in history]


# def get_bot_response(message, history, llm):
#    messages = [
#        SystemMessage(content="You are a helpful assistant."),
#    ]
#    for msg, is_user in history:
#        if is_user:
#            messages.append(HumanMessage(content=msg))
#        else:
#            messages.append(AIMessage(content=msg))
#    messages.append(HumanMessage(content=message))

#    response = llm(messages)
#    return response.content


def create_conversation(
    user_id,
    use_user_preferences,
    title=None,
):
    if title is None:
        title = "New Chat"
    encrypted_title = encrypt_data(title)
    c.execute(
        "INSERT INTO conversations (user_id, title, use_user_preferences) VALUES (?, ?, ?)",
        (user_id, encrypted_title, use_user_preferences),
    )
    conn.commit()
    return c.lastrowid


def update_conversation_title(conversation_id, title):
    encrypted_title = encrypt_data(title)
    c.execute(
        "UPDATE conversations SET title = ? WHERE id = ?",
        (encrypted_title, conversation_id),
    )
    conn.commit()


def get_user_conversations(user_id):
    c.execute(
        """
        SELECT id, title, use_user_preferences, created_at 
        FROM conversations 
        WHERE user_id = ? 
        ORDER BY created_at DESC""",
        (user_id,),
    )
    conversations = c.fetchall()
    # 解密标题
    return [
        (id, decrypt_data(title), use_user_preferences, created_at)
        for id, title, use_user_preferences, created_at in conversations
    ]


def is_first_message(conversation_id):
    c.execute(
        """SELECT COUNT(*) FROM chat_history 
                 WHERE conversation_id = ? AND is_user = 1""",
        (conversation_id,),
    )
    count = c.fetchone()[0]
    return count == 0


def save_message(user_id, conversation_id, message, is_user, sentiment_score):
    if is_user and is_first_message(conversation_id):
        title = message[:50] if len(message) > 50 else message
        update_conversation_title(conversation_id, title)
        # first message will not consider as user feedback
        sentiment_score = 0

    encrypted_message = encrypt_data(message)
    c.execute(
        """
        INSERT INTO chat_history (user_id, conversation_id, message, is_user, sentiment_score) 
        VALUES (?, ?, ?, ?, ?)""",
        (user_id, conversation_id, encrypted_message, is_user, sentiment_score),
    )
    conn.commit()


def load_chat_history(user_id, conversation_id):
    c.execute(
        """
        SELECT message, is_user 
        FROM chat_history 
        WHERE user_id = ? AND conversation_id = ? 
        ORDER BY timestamp""",
        (user_id, conversation_id),
    )
    history = c.fetchall()
    return [(decrypt_data(msg), is_user) for msg, is_user in history]


# Define a list of preferences
user_preferences = [
    "Reading",
    "Traveling",
    "Cooking",
    "Gaming",
    "Fitness",
    "Music",
    "Art",
    "Technology",
    "Gardening",
    "Photography",
]


def login_page():
    st.title("Login")

    # Initialize session state for the button - login or register
    if "button_clicked_4_login" not in st.session_state:
        st.session_state.button_clicked_4_login = False
    if "button_clicked_4_register" not in st.session_state:
        st.session_state.button_clicked_4_register = False

    if "consent_checkbox" not in st.session_state:
        st.session_state.consent_checkbox = False
    if "consent_warning" not in st.session_state:
        st.session_state.consent_warning = False

    # Show the button only if it hasn't been clicked yet
    if (
        not st.session_state.button_clicked_4_login
        and not st.session_state.button_clicked_4_register
    ):
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Login"):
                st.session_state.button_clicked_4_login = True
                st.rerun()
        with col2:
            if st.button("Register"):
                st.session_state.button_clicked_4_register = True
                st.rerun()

    if (
        st.session_state.button_clicked_4_login == True
        or st.session_state.button_clicked_4_register == True
    ):
        username = st.text_input("Username", value="")
        password = st.text_input("Password", value="", type="password")
        if st.session_state.button_clicked_4_login == True:
            if st.button("Login"):
                user_id = verify_user(username, password)
                if user_id:
                    st.session_state.user_id = user_id
                    st.session_state.username = username
                    st.success("Successfully Login!")
                    st.rerun()
                else:
                    st.error("Incorrect Username or Password")
        if st.session_state.button_clicked_4_register == True:
            # Create checkboxes based on the list of preferences
            st.write("**Select your preferences**")
            selected_preferences = []
            for preference in user_preferences:
                if st.checkbox(preference):
                    selected_preferences.append(preference)
            others_preference = st.text_area("Enter your preference here:")
            st.write(
                """**Consent Statement**
We value your privacy and are committed to protecting your personal data. By ticking the box below and clicking 'Register', you consent to our collection, use, and storage of your personal data in accordance with our Privacy Policy. This data will be used solely for the purpose of [specify purpose, e.g., improving our services, sending newsletters, etc.]. You have the right to withdraw your consent at any time by contacting us at [contact information]."""
            )
            st.session_state.consent_checkbox = st.checkbox(
                "I agree to the collection, use, and storage of my personal data as described above."
            )
            if st.button("Register"):
                if not st.session_state["consent_checkbox"]:
                    st.session_state["consent_warning"] = True
                else:
                    st.session_state["consent_warning"] = False

                if st.session_state["consent_warning"]:
                    st.markdown(
                        "<span style='color: red;'>You must agree to the Consent Statement!</span>",
                        unsafe_allow_html=True,
                    )
                else:
                    if others_preference.strip() != "":
                        selected_preferences.append(others_preference)
                    if create_user(username, password, selected_preferences):
                        st.session_state.button_clicked_4_login = True
                        st.session_state.button_clicked_4_register = False
                        del st.session_state.consent_checkbox
                        del st.session_state.consent_warning
                        st.rerun()
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
                st.session_state["model_map"][option] = torch.load("your path")
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
    if "current_conversation_use_user_preferences" not in st.session_state:
        st.session_state.current_conversation_use_user_preferences = False

    if "ui" not in st.session_state:
        session_state = SessionState()
        session_state.init_state()
        st.session_state["ui"] = StreamlitUI(session_state)

    with st.sidebar:
        if st.button("Logout"):
            del st.session_state.user_id
            del st.session_state.username
            del st.session_state.button_clicked_4_login
            del st.session_state.button_clicked_4_register
            del st.session_state.current_conversation_use_user_preferences
            del st.session_state.current_conversation_id
            st.rerun()

        st.write("Create New Chat")

        # fix: change the display of "system_prompt_setting" and "temperature_setting"
        if "current_conversation_id" not in st.session_state:
            # use default value
            current_settings = get_conversation_settings(-1)
        else:
            # load base on st.session_state.current_conversation_id
            current_settings = get_conversation_settings(
                st.session_state.current_conversation_id
            )

        system_prompt_setting = current_settings["system_prompt"]
        temperature_setting = float(current_settings["temperature"])

        system_prompt = st.text_area(
            "Set up AI persona (System Prompt):",
            value=system_prompt_setting,
            help="Set the behavior and personality traits of AI assistants",
        )

        temperature = st.slider(
            "Answer randomness (Temperature):",
            min_value=0.0,
            max_value=1.0,
            value=temperature_setting,
            help="A higher value of randomness in the answer will make the answer more creative, while a lower value will make the answer more certain.",
        )

        col1, col2 = st.columns([1, 2])
        with col1:
            create_new_chat = st.button("Create New Chat")
        with col2:
            use_user_preferences_selection = st.checkbox("(with user preferences)")

        if create_new_chat:
            new_conv_id = create_conversation(
                st.session_state.user_id, use_user_preferences_selection
            )
            save_conversation_settings(new_conv_id, system_prompt, temperature)
            st.session_state.current_conversation_id = new_conv_id
            st.session_state.current_conversation_use_user_preferences = (
                use_user_preferences_selection
            )
            print("use_user_preferences_selection")
            print(use_user_preferences_selection)

            print("st.session_state.current_conversation_use_user_preferences")
            print(st.session_state.current_conversation_use_user_preferences)
            st.rerun()

        conversations = get_user_conversations(st.session_state.user_id)
        if conversations:
            st.write("Select an Existing Chat:")
            for conv_id, title, use_user_preferences, created_at in conversations:
                col1, col2 = st.columns([1, 1])
                with col1:
                    if st.button(f"{title} ({created_at})", key=f"conv_{conv_id}"):
                        st.session_state.current_conversation_id = conv_id
                        st.session_state.current_conversation_use_user_preferences = (
                            use_user_preferences
                        )
                        st.rerun()
                with col2:
                    if st.button("Delete", key=f"del_{conv_id}"):
                        delete_conversation(conv_id)
                        if (
                            "current_conversation_id" in st.session_state
                            and st.session_state.current_conversation_id == conv_id
                        ):
                            del st.session_state.current_conversation_id
                        st.rerun()

    if "current_conversation_id" not in st.session_state:
        st.info("Please create a new chat or select an existing chat")
        return

    current_settings = get_conversation_settings(
        st.session_state.current_conversation_id
    )
    # update current_settings: mainly if a new chat has been created
    system_prompt_setting = current_settings["system_prompt"]
    temperature_setting = current_settings["temperature"]

    model_name, uploaded_file = st.session_state["ui"].setup_sidebar()

    if model_name == "Doubao-Pro-32k":
        inference_server_url = "https://ark.cn-beijing.volces.com/api/v3"
        api_key = "72c991d1-8078-4c49-8b5e-7b33c4e26e04"
        model_name = "ep-20240722030254-mrqhx"

        llm = ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            openai_api_base=inference_server_url,
            temperature=temperature_setting,
        )

    # if user_perferences is not empty, we append preferences information as prefix
    def get_bot_response(message, history, llm, system_prompt, user_perferences):
        messages = [
            SystemMessage(content=system_prompt),
        ]
        user_first_msg_prefix = ""
        if len(user_perferences) > 0:
            topic = ""
            for user_id, perferences in user_perferences:
                topic += perferences + ","
            # remove last char
            topic = topic[:-1]
            user_first_msg_prefix = (
                "I am interested in:"
                + topic
                + ". Provide information base of my interested topic."
            )
            print(user_first_msg_prefix)
        isFirstMessage = True
        for msg, is_user in history:
            if is_user:
                if isFirstMessage:
                    # history will not keep "user_first_msg_prefix" -> that is the interest
                    messages.append(HumanMessage(content=user_first_msg_prefix + msg))
                    isFirstMessage = False
                else:
                    messages.append(HumanMessage(content=msg))
            else:
                messages.append(AIMessage(content=msg))
        if isFirstMessage:
            # history will not keep "user_first_msg_prefix" -> that is the interest
            messages.append(HumanMessage(content=user_first_msg_prefix + message))
        else:
            messages.append(HumanMessage(content=message))

        response = llm(messages)
        return response.content

    history = load_chat_history(
        st.session_state.user_id, st.session_state.current_conversation_id
    )
    for message, is_user in history:
        if is_user:
            st.session_state["ui"].render_user(message)
        else:
            st.session_state["ui"].render_assistant(message)

    user_input = st.chat_input("Input your message:")

    if user_input:
        # get_sentiment_score for user response
        sentiment_score = get_sentiment_score(user_input)
        response_prefix = get_response_prefix(sentiment_score)

        save_message(
            st.session_state.user_id,
            st.session_state.current_conversation_id,
            user_input,
            True,
            sentiment_score,
        )
        st.session_state["ui"].render_user(user_input)

        # construct the searched link and display to user
        result_link = ""
        # some keyword that will trigger the search function
        search_keyword_list = ["search", "find", "what"]
        if any(substring in user_input.lower() for substring in search_keyword_list):
            search_results = google_search(user_input)
            search_summary = search_results.get("items", [])
            web_summary = ""
            # prepare part of the response that append at the response.
            result_link = "You can find more infomration in: "
            for result in search_summary:
                web_content = fetch_web_content(result["link"])
                # some response may not be 200
                if web_content is not None:
                    # print("Web Content:" + web_content)
                    # get summarized_content
                    summarized_content = summarize_content(web_content)
                    # prepare content that will use as part of user input
                    web_summary += summarized_content
                    result_link += result["link"] + "; "
                    # print("Link:" + result["link"])
                    # print("summarized_content:" + summarized_content)

            # user_input += "In additional, please also base on the following information to provide information: " + web_summary
            # tell LLM that it should provide information base on summary
            user_input += (
                "Consider the following content to provide information: " + web_summary
            )

        if st.session_state.current_conversation_use_user_preferences == True:
            user_preference = get_user_preference(st.session_state.user_id)
        else:
            user_preference = []

        bot_response = get_bot_response(
            user_input, history, llm, system_prompt_setting, user_preference
        )

        bot_response_with_response_prefix = (
            response_prefix.strip() + bot_response + result_link.strip()
        )

        save_message(
            st.session_state.user_id,
            st.session_state.current_conversation_id,
            bot_response_with_response_prefix,
            False,
            0,
        )
        st.session_state["ui"].render_assistant(bot_response_with_response_prefix)
        st.rerun()


def get_sentiment_score(text):
    return sentiment_analyzer.polarity_scores(text)["compound"]


# for web search function
google_search_url = "https://www.googleapis.com/customsearch/v1"
google_api_key = "AIzaSyADcVI9ooWqFzbvp0SHeWSIR3GCu-6nwEo"
search_engine_id = "41100463dc2ae4b63"  # search for every date is limited


def google_search(query):
    headers = {"Content-Type": "application/json"}
    params = {
        "key": google_api_key,
        "cx": search_engine_id,
        "q": query,
        "num": 3,  # Limit to top 3 results
    }
    response = requests.get(google_search_url, headers=headers, params=params)
    return response.json()


def get_google_search_respone(query):
    # Perform a web search for additional information
    search_results = google_search(query)
    search_summary = search_results.get("items", [])
    return search_summary


def get_response_prefix(sentiment_score):
    response_prefix = ""
    if sentiment_score < -0.5:
        response_prefix = "I'm sorry to hear that. "
    elif sentiment_score > 0.5:
        response_prefix = "That's great! "
    else:
        response_prefix = ""
    return response_prefix


# get content from url
def fetch_web_content(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text()
    else:
        return None


#
import nltk
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

# Ensure the punkt tokenizer is downloaded
# for safe: python -m nltk.downloader all
nltk.download("punkt")


# smmary the content
def summarize_content(content, num_sentences=2):
    # print('***summarize_content***' + content)
    parser = PlaintextParser.from_string(content, Tokenizer("english"))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, 2)  # Summarize into 2 sentences
    summary_text = " ".join([str(sentence) for sentence in summary])
    summary_sentences = nltk.tokenize.sent_tokenize(summary_text)
    print(summary_sentences)
    return " ".join(summary_sentences[:num_sentences])


def main():
    if "user_id" not in st.session_state:
        login_page()
    else:
        chat_page()


if __name__ == "__main__":
    main()
