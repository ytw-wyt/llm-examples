import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import os

st.write(st.session_state.generated_course_4)
# Navigation buttons
# col1, col2 = st.columns([5,1])

# with col1:
#     if st.button("Previous"): 
#         switch_page("step3 result3")

def save_page():

    # Define the page content
    page_content = f"""
import streamlit as st

def main():
    st.title("Generated Courses")

    st.write("Course 1:")
    st.write(\"\"\"{st.session_state.generated_course_1}\"\"\")

    st.write("Course 2:")
    st.write(\"\"\"{st.session_state.generated_course_2}\"\"\")

    st.write("Course 3:")
    st.write(\"\"\"{st.session_state.generated_course_3}\"\"\")

    st.write("Course 4:")
    st.write(\"\"\"{st.session_state.generated_course_4}\"\"\")

if __name__ == "__main__":
    main()
"""

    # Ensure the pages directory exists
    if not os.path.exists('pages'):
        os.makedirs('pages')

    # Write the page content to a new Python file
    with open('pages/generated_page.py', 'w') as f:
        f.write(page_content)

    st.success("Page generated and saved successfully!")


col1, col2, col3 = st.columns([1, 5, 1])

with col1:
    if st.button("Previous"):
        pass

# with col2:
#     st.download_button(
#         label="Save Results",
#         data=f"Course 1:\n{st.session_state.generated_course_1}\n\nCourse 2:\n{st.session_state.generated_course_2}\n\nCourse 3:\n{st.session_state.generated_course_3}\n\nCourse 4:\n{st.session_state.generated_course_4}",
#         file_name="generated_courses.txt",
#         mime="text/plain"
#     )
with col2:
        if st.button("Save Page"):
            save_page()

