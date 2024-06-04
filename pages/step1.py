import streamlit as st
import anthropic
from streamlit_extras.switch_page_button import switch_page

if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

with st.sidebar:
    anthropic_api_key = st.text_input("Anthropic API Key", key="file_qa_api_key", type="password")
    "[View the source code](https://github.com/streamlit/llm-examples/blob/main/pages/1_File_Q%26A.py)"
    "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.text("What are you looking to learn today?")
st.header("Input your customized course topic")

current_step = st.progress(50)

col1, col2 = st.columns([1,1])
with col1:
    st.page_link("pages/step1.py", label="State your learning goal", icon="1️⃣")

with col2:
    st.page_link("pages/step2.py", label="Upload your research papers", icon="2️⃣", disabled=True)

# Initialize session state if it does not exist
if 'topic' not in st.session_state:
    st.session_state.topic = None

if 'learning_objective' not in st.session_state:
    st.session_state.learning_objective = ""

st.session_state.topic = st.selectbox(
    "Select a topic",
    (
        "Social Emotional Learning", 
        "Mastery of Content", 
        "Advocary", 
        "Building Relationships", 
        "Utilizing Technology Tools", 
        "Domain Specific Knowledge"
    ),
    index=None if st.session_state.topic is None else (
        ["Social Emotional Learning", "Mastery of Content", "Advocary", "Building Relationships", "Utilizing Technology Tools", "Domain Specific Knowledge"].index(st.session_state.topic)
    )
)
# Text input for learning objective
st.session_state.learning_objective = st.text_input(
    "What's your learning objective?",
    value=st.session_state.learning_objective,
    label_visibility="visible",
    disabled=False,
    placeholder="Enter your objective here..."
)
# text_input = st.text_input(
#         "What's your learning objective?",
#         label_visibility=st.session_state.visibility,
#         disabled=st.session_state.disabled,
#         # placeholder=st.session_state.placeholder,
#     ),

col1, col2 = st.columns([5,1])
with col1:
    if st.button("Back"): pass

with col2:
    if st.button("Next"):
        switch_page("step2")