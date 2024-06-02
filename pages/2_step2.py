import streamlit as st
from streamlit_extras.switch_page_button import switch_page

st.text("What are you looking to learn today?")
st.header("Input your customized course topic")

current_step = st.progress(100)

col1, col2 = st.columns([1,1])
with col1:
    st.page_link("pages/1_step1.py", label="State your learning goal", icon="1️⃣")

with col2:
    st.page_link("pages/2_step2.py", label="Upload your research papers", icon="2️⃣")


uploaded_file = st.file_uploader("Upload an article", type=("txt", "md"))
question = st.text_input(
    "Ask something about the article",
    placeholder="Can you give me a short summary?",
    disabled=not uploaded_file,
)

if uploaded_file and question and not anthropic_api_key:
    st.info("Please add your Anthropic API key to continue.")

if uploaded_file and question and anthropic_api_key:
    article = uploaded_file.read().decode()
    prompt = f"""{anthropic.HUMAN_PROMPT} Here's an article:\n\n<article>
    {article}\n\n</article>\n\n{question}{anthropic.AI_PROMPT}"""

    client = anthropic.Client(api_key=anthropic_api_key)
    response = client.completions.create(
        prompt=prompt,
        stop_sequences=[anthropic.HUMAN_PROMPT],
        model="claude-v1",  # "claude-2" for Claude 2 model
        max_tokens_to_sample=100,
    )
    st.write("### Answer")
    st.write(response.completion)

col1, col2 = st.columns([5,1])
with col1:
    if st.button("Back"): 
        switch_page("step1")

with col2:
    if st.button("Next"):
        switch_page("step3")