import streamlit as st
from streamlit_chat import message
from langchain.embeddings import OpenAIEmbeddings
from openai.embeddings_utils import cosine_similarity
import os
from utils import get_word_wordnet, find_synonyms
from dotenv import load_dotenv
load_dotenv()


st.set_page_config(page_title='Word Game', page_icon=":robot_face:")
st.markdown("<h2 style='text-align: center;'> Guess the word </h2>", unsafe_allow_html=True)

# Initialize session state to store text entries
if "text_entries" not in st.session_state:
    st.session_state.text_entries = []

# Initialize session state to store randon word
if "random_word" not in st.session_state:
    st.session_state.random_word = get_word_wordnet()

# Initialize session state to store hints
if "hints" not in st.session_state:
    st.session_state.hints = []

# Initialize session state to store hints
if "quit" not in st.session_state:
    st.session_state.quit = ""

# implement container feature
main_container = st.container()

# Custom CSS for a container with text at the bottom
custom_css = """
<style>
.container-with-bottom-text {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 50px;
    font-size: 300px;
    font-weight: bold;
}
</style>
"""
# Inject the CSS
st.markdown(custom_css, unsafe_allow_html=True)

# CSS for text

st.markdown("""
    <style>
    .increase-font-size {
        font-size: 16px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)


# A container for user input text box
sub_container = st.container()
sub_container.markdown(custom_css, unsafe_allow_html=True)


with main_container:

    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_input("Enter word here: ", key='input')
        submit_button = st.form_submit_button(label='Enter')

        if submit_button:
            if user_input.strip():  # Add only non-empty inputs
                #convert user input to lower case
                user_input = user_input.lower()
                st.session_state.text_entries.append(user_input)

    if user_input.lower() == st.session_state.random_word:
            st.markdown(f"""
                        <p style="color: green; font-size: 16px; font-weight: bold; text-align: left;">You guessed correctly!!! The word is: {st.session_state.random_word}</p>
                    """, unsafe_allow_html=True)
    with st.container():
        # Create two columns
        col1, col2 = st.columns([2, 10])  # The values inside the list represent the width ratio of the columns
        with col1:
            if st.button("Hint: "):
                hint = find_synonyms(st.session_state.random_word, st.session_state.hints)
                st.session_state.hints.append(hint)
                  
        with col2:
                if(len(st.session_state.hints) > 0):
                    index = len(st.session_state.hints) - 1
                    st.markdown(f"""
                        <p style="color: #40E0D0; font-size: 16px; font-weight: bold; text-align: left;">{st.session_state.hints[index]}</p>
                    """, unsafe_allow_html=True)

        col3, col4 = st.columns([2, 10])  # The values inside the list represent the width ratio of the columns
        with col3:
            if st.button("Quit: "):
                    st.session_state.quit = "clicked"
        with col4:
                if st.session_state.quit == "clicked":   
                    st.markdown(f"""
                            <p style="color: red; font-size: 16px; font-weight: bold; text-align: left;"> The word was: {st.session_state.random_word}</p>
                        """, unsafe_allow_html=True)

    if st.session_state.text_entries:

        for entry in reversed(st.session_state.text_entries):  # Show newest first

            print(f"Generated session word: {st.session_state.random_word}")

            # Initialize the OpenAIEmbeddings object
            embeddings = OpenAIEmbeddings()
            embed_word = embeddings.embed_query(st.session_state.random_word)

            text_embedding = embeddings.embed_query(entry)
            similarity_score = cosine_similarity(embed_word, text_embedding)


            with sub_container:  # Sub-container for each entry
                   # Create two columns within the container
                col1, col2 = st.columns(2)

                with col1:
                    # Use the dynamic text variable in the HTML
                    st.markdown(f"""
                    <div class="container-with-bottom-text">
                        <p class="increase-font-size">{entry}</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    # Add custom-styled bar with percentage
                    custom_css_2 = """
                    <style>
                    .custom-bar-container {
                        position: absolute;
                        height: 35px;
                        width: 100%;
                        background-color: #f3f3f3;
                        border: 1px solid #ccc;
                        border-radius: 5px;
                    }

                    .custom-bar {
                        height: 100%;
                        background-color: #4caf50;
                        width: 0%;
                        transition: width 0.5s;
                        border-radius: 5px;
                    }

                    .progress-text {
                        position: absolute;
                        top: 50%;
                        left: 50%;
                        transform: translate(-50%, -50%);
                        font-weight: bold;
                    }
                    </style>
                    """

                    st.markdown(custom_css_2, unsafe_allow_html=True)

                    def render_custom_bar(percent):
                        st.markdown(f"""
                        <div class="custom-bar-container">
                            <div class="custom-bar" style="width: {percent}%;"></div>
                            <div class="progress-text">{percent}%</div>
                        </div>
                        """, unsafe_allow_html=True)

                    similarity_score = similarity_score * 100
                    print(f"Smilarity score: {similarity_score}")
                    render_custom_bar(round(similarity_score, 2))  # Update the custom bar

    else:
        st.write("No entries yet. Add some using the text box above!")

# Add some vertical space for better aesthetics
st.markdown("<style>div.stContainer{max-height:400px;overflow:auto;}</style>", unsafe_allow_html=True)
