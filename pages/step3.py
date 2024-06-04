import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import time
import streamlit.components.v1 as components
from pages.step2 import generate_course_section
def clear_cache():
    st.session_state.generated_course_1 = None
    st.session_state.generated_course_2 = None
    st.session_state.generated_course_3 = None
    st.session_state.generated_course_4 = None

if 'generated_course_1' not in st.session_state:
    st.session_state.generated_course_1 = None
if 'generated_course_2' not in st.session_state:
    st.session_state.generated_course_2 = None
if 'generated_course_3' not in st.session_state:
    st.session_state.generated_course_3 = None
if 'generated_course_4' not in st.session_state:
    st.session_state.generated_course_4 = None

def main():
    clear_cache()
    if st.session_state.generated_course_1 is None:
        with st.spinner("We have all we needed. Loading the result."):
            time.sleep(3)
    result_1 = generate_course_section(st.session_state.key, st.session_state.text, st.session_state.template1)
    result_2 = generate_course_section(st.session_state.key, st.session_state.text, st.session_state.template2)
    result_3 = generate_course_section(st.session_state.key, st.session_state.text, st.session_state.template3)
    result_4 = generate_course_section(st.session_state.key, st.session_state.text, st.session_state.template4)
    st.session_state.generated_course_1 = result_1
    st.session_state.generated_course_2 = result_2
    st.session_state.generated_course_3 = result_3
    st.session_state.generated_course_4 = result_4

    st.write(st.session_state.generated_course_1)

    # Navigation buttons
    col1, col2 = st.columns([5,1])

    with col1:
        if st.button("Previous"): pass

    with col2:
        if st.button("Next"):
            switch_page("step3 result2")

if __name__ == "__main__":
    main()
