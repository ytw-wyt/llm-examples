import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import pprint

st.text("What are you looking to learn today?")
st.header("Input your customized course topic")

current_step = st.progress(100)

col1, col2 = st.columns([1,1])
with col1:
    st.page_link("pages/1_step1.py", label="State your learning goal", icon="1️⃣")

with col2:
    st.page_link("pages/2_step2.py", label="Upload your research papers", icon="2️⃣")

uploaded_file = st.file_uploader("Upload the articles you want to reference. (You can upload up to 5 files)", type=("txt", "md"))
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




# openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")


# def env_setup(context, question):
#     # Instantiate LLM model
#     llm = OpenAI(model_name="text-davinci-003", openai_api_key=openai_api_key)
#     # Prompt
#     template = prompt_template = """You are an expert course generator tasked with creating a comprehensive tutor training course for online math tutors. Use the following pieces of context to answer the question and following the template the user gives you, but you should not refer to the template context, and please generate a course to the user based on the research paper. The users of the course are novice tutors who are experts on the math subjects that they are teaching, but unfamiliar with the best method of teaching what they know to the students. The course should have a course title
#     and some practical examples (each scenario should have 4 questions) and some practical research recommendation examples for the tutor to show how they could perform in their classrooms.

#     If you don't know the answer, just say "I don't know".

#     {context}

#     Question: {question}
#     """
#     PROMPT = PromptTemplate(input_variables=['context', 'question'], template=template)
#     prompt_query = prompt.format(topic=topic)
#     # Create the Chain for the chat with the retriever and the prompt template
#     qachain = RetrievalQA.from_chain_type(llm=ChatOpenAI(
#                                     model_name = "gpt-4o",
#                                     temperature=0.6,
#                                     verbose=False),
#                                     chain_type='stuff',
#                                     chain_type_kwargs={'prompt': PROMPT},
#                                     retriever=custom_retriever)


#     response = llm(prompt_query)
#     # Print results
#     return st.info(response)

# def topic_learning_objective(context, question):
#     template = """
#     Do you know what math topic the paper is talking about? Can you generate a scenario-based tutor training course about how to teach the mathematic topic effectively as discussed in the retrieved research paper?

#     I need your help to generate the course title, description and learning objectives, please follow the below template.

#     Course Title: Generate a title of this math course using three words, the title should begin with a verb. It should be related to the specific math topic of the research paper and the course objective.
#     Example titles are: Using polite language, Managing inequity, Managing effective praise


#     Description: A short description about the purpose of this course and why. The structure could be similar to :  Have you ever met a situation where you want to teach [the math topic] but you find yourself unable to explain the concept clearly to the students? In this module, we will be introducing [strategy name] as a way of tutoring students about [the math topic].


#     Learning Objectives:
#     Requirement for generation: The learning objectives should address the "understanding" and the "creating" level of Bloom's taxonomy.
#     Creating meanse use information to create something new, understanding means grasp meaning of instructional materials.

#     Objective 1: Describe the expected outcome concerning the first objective.
#     Objective 2: Outline what learners will achieve by the end of this module regarding the second objective.
#     Add additional objectives as necessary.

#     One of the objectives should clearly state the strategy that is advocated in the retrieved research paper.

#     You don't need to generate the specific scenario at this time.
#     """

#     doc_prompt = qachain({"query": template})
#     pprint.pprint(doc_prompt['result'])

col1, col2 = st.columns([5,1])
with col1:
    if st.button("Back"): 
        switch_page("step1")

with col2:
    if st.button("Next"):
        switch_page("step3")