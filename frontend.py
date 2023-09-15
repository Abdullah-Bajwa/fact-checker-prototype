import streamlit as st
from fact_checker_via_web import FactCheckerViaWeb

# Create an instance of FactCheckerViaWeb
fact_checker = FactCheckerViaWeb()

# Customize Streamlit app configuration to remove the Streamlit footer
st.set_page_config(page_title="CyberNator Fact Check", page_icon=":pencil:", layout="wide")

# Streamlit app title and description
st.title("CyberNator Fact Checker")
st.write("Enter text and click 'Submit' to fact-check.")

# Create a text input widget
user_input = st.text_area("Enter text here:")

# Function to fact-check the text
def fact_check_text(input_text):
    text, rewrite = fact_checker.fact_check_sentence(input_text, None)
    return text, rewrite

# Create a "Submit" button
if st.button("Submit"):
    if user_input:
        # Call the fact_check_text function with the user's input
        original_sentence, suggested_rewrite = fact_check_text(user_input)

        # Display the original sentence with marked errors and suggested rewrites as HTML using st.markdown
        st.subheader("Original Sentence with Marked Errors:")
        st.markdown(original_sentence, unsafe_allow_html=True)  # Allow rendering HTML

        st.subheader("Suggested Rewrites:")
        st.markdown(suggested_rewrite, unsafe_allow_html=True)  # Allow rendering HTML

# Provide some instructions to the user
st.write("Note: You can enter text in the input box above and click 'Submit' to perform the fact-check.")

# Improve the overall app layout and styling
#st.sidebar.title("Settings")
#st.sidebar.write("Customize your experience here.")

# Add some padding and background color to the main content
st.markdown(
    """
    <style>
    .st-dg {
        padding: 20px;
        background-color: #f5f5f5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
