import streamlit as st
from pathlib import Path

from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain import OpenAI
from langchain.llms.openai import OpenAIChat
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath('data', 'vector_store')

st.set_page_config(page_title="RAG")
st.title("Retrieval Augmented Generation Engine")

def load_documents():
    loader = DirectoryLoader(TMP_DIR.as_posix(), glob='**/*.pdf')
    documents = loader.load()
    return documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=100, separators=["\n\n", "\n\n\n", "\n", " ", ""]
    )
    text = text_splitter.split_documents(documents)
    return texts

def embeddings_on_local_vectordb(texts):
    persist_directory = 'chroma_db'
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(documents=docs, embedding=embeddings, persist_directory=persist_directory)
    vectordb.persist()
    vector_db = None
    vector_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    custom_retriever = vector_db.as_retriever(search_type="similarity_score_threshold",
                                          search_kwargs={"k": 5,
                                          "score_threshold":0.4})
    custom_retriever = vector_db.as_retriever(search_type="mmr",
                                          search_kwargs={"fetchK": 10, "lambda": 0.25})
    return custom_retriever

def env_setup(context, question):
    # Instantiate LLM model
    llm = OpenAI(model_name="text-davinci-003", openai_api_key=openai_api_key)
    # Prompt
    template = prompt_template = """You are an expert course generator tasked with creating a comprehensive tutor training course for online math tutors. Use the following pieces of context to answer the question and following the template the user gives you, but you should not refer to the template context, and please generate a course to the user based on the research paper. The users of the course are novice tutors who are experts on the math subjects that they are teaching, but unfamiliar with the best method of teaching what they know to the students. The course should have a course title
    and some practical examples (each scenario should have 4 questions) and some practical research recommendation examples for the tutor to show how they could perform in their classrooms.

    If you don't know the answer, just say "I don't know".

    {context}

    Question: {question}
    """
    PROMPT = PromptTemplate(input_variables=['context', 'question'], template=template)
    prompt_query = prompt.format(topic=topic)
    # Create the Chain for the chat with the retriever and the prompt template
    qachain = RetrievalQA.from_chain_type(llm=ChatOpenAI(
                                    model_name = "gpt-4o",
                                    temperature=0.6,
                                    verbose=False),
                                    chain_type='stuff',
                                    chain_type_kwargs={'prompt': PROMPT},
                                    retriever=custom_retriever)

def topic_learning_objective(context, question):
    template = """
    Do you know what math topic the paper is talking about? Can you generate a scenario-based tutor training course about how to teach the mathematic topic effectively as discussed in the retrieved research paper?

    I need your help to generate the course title, description and learning objectives, please follow the below template.

    Course Title: Generate a title of this math course using three words, the title should begin with a verb. It should be related to the specific math topic of the research paper and the course objective.
    Example titles are: Using polite language, Managing inequity, Managing effective praise


    Description: A short description about the purpose of this course and why. The structure could be similar to :  Have you ever met a situation where you want to teach [the math topic] but you find yourself unable to explain the concept clearly to the students? In this module, we will be introducing [strategy name] as a way of tutoring students about [the math topic].


    Learning Objectives:
    Requirement for generation: The learning objectives should address the "understanding" and the "creating" level of Bloom's taxonomy.
    Creating meanse use information to create something new, understanding means grasp meaning of instructional materials.

    Objective 1: Describe the expected outcome concerning the first objective.
    Objective 2: Outline what learners will achieve by the end of this module regarding the second objective.
    Add additional objectives as necessary.

    One of the objectives should clearly state the strategy that is advocated in the retrieved research paper.

    You don't need to generate the specific scenario at this time.
    """

    doc_prompt = qachain({"query": template})
    pprint.pprint(doc_prompt['result'])

def input_fields():
    st.session_state.source_docs = st.file_uploader(label="Upload Documents", type="pdf", accept_multiple_files=True)


def process_documents():
    if not st.session_state.openai_api_key or not st.session_state.pinecone_api_key or not st.session_state.pinecone_env or not st.session_state.pinecone_index or not st.session_state.source_docs:
        st.warning(f"Please upload the documents and provide the missing fields.")
    else:
        try:
            for source_doc in st.session_state.source_docs:
                #
                with tempfile.NamedTemporaryFile(delete=False, dir=TMP_DIR.as_posix(), suffix='.pdf') as tmp_file:
                    tmp_file.write(source_doc.read())
                #
                documents = load_documents()
                #
                for _file in TMP_DIR.iterdir():
                    temp_file = TMP_DIR.joinpath(_file)
                    temp_file.unlink()
                #
                texts = split_documents(documents)
                #
                st.session_state.retriever = embeddings_on_local_vectordb(texts)
        except Exception as e:
            st.error(f"An error occurred: {e}")

def boot():
    #
    input_fields()
    #
    st.button("Submit Documents", on_click=process_documents)
    #
    if "messages" not in st.session_state:
        st.session_state.messages = []
    #
    for message in st.session_state.messages:
        st.chat_message('human').write(message[0])
        st.chat_message('ai').write(message[1])    
    #
    if query := st.chat_input():
        st.chat_message("human").write(query)
        response = query_llm(st.session_state.retriever, query)
        st.chat_message("ai").write(response)

if __name__ == '__main__':
    #
    boot()