import streamlit as st
import anthropic
from streamlit_extras.switch_page_button import switch_page

if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

# with st.sidebar:
#     anthropic_api_key = st.text_input("Anthropic API Key", key="file_qa_api_key", type="password")
#     "[View the source code](https://github.com/streamlit/llm-examples/blob/main/pages/1_File_Q%26A.py)"
#     "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

st.header("What course topic do you want to generate for today?")
st.header("Input your customized course topic")

current_step = st.progress(50)
# Initialize session state if it does not exist
if 'topic' not in st.session_state:
    st.session_state.topic = None

if 'learning_objective' not in st.session_state:
    st.session_state.learning_objective = ""

col1, col2 = st.columns([1,1])
with col1:
    st.page_link("step1.py", label="State your learning goal", icon="1️⃣")

if st.session_state.topic and st.session_state.learning_objective:
    with col2:
        st.page_link("pages/step2.py", label="Upload your research papers", icon="2️⃣", disabled=False)
else:
    with col2:
        st.page_link("pages/step2.py", label="Upload your research papers", icon="2️⃣", disabled=True)

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

col1, col2 = st.columns([5,1])

if st.session_state.topic and st.session_state.learning_objective:
    with col2:
        if st.button("Next"):
            switch_page("step2")
else:
    with col2:
        st.button("Next", disabled=True)