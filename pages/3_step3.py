import streamlit as st
if 'generated_course' in st.session_state:
    st.info("🤖 Generated Course: " + st.session_state.generated_course)
else:
    st.header("We have all we needed. Loading the result.")