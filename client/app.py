import streamlit as st
import random
import sys
import os
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Now we can import from models directory
try:
    from models.cece import RAGApplication
except ImportError as e:
    print(f"Error importing RAGApplication: {e}")
    
# Function to initialize RAG
@st.cache_resource
def init_rag():
    try:
        file_paths = ["textbook.pdf"]  # Update with your actual file path
        openai_api_key = os.getenv("OPENAI_KEY")  # Make sure to set this in your environment
        return RAGApplication(
            file_paths=file_paths,
            openai_api_key=openai_api_key
        )
    except Exception as e:
        st.error(f"Error initializing RAG: {str(e)}")
        return None

# Function to generate fixed class data
def generate_fixed_classes():
    return [
        "CS301: Machine Learning",
        "AI401: Deep Learning",
        "DS201: Computer Vision",
        "CS501: Natural Language Processing"
    ]

# Function to generate fake quizzes for a class
def generate_fake_quizzes(class_name):
    quiz_topics = {
        "CS301: Machine Learning": ["Linear Regression", "Decision Trees", "Support Vector Machines", "Clustering"],
        "AI401: Deep Learning": ["Neural Networks", "Backpropagation", "Convolutional Networks", "Recurrent Networks"],
        "DS201: Computer Vision": ["Image Processing", "Feature Detection", "Object Recognition", "Segmentation"],
        "CS501: Natural Language Processing": ["Tokenization", "Part-of-Speech Tagging", "Named Entity Recognition", "Sentiment Analysis"]
    }
    
    topics = quiz_topics.get(class_name, ["General Topic 1", "General Topic 2", "General Topic 3", "General Topic 4"])
    selected_topics = random.sample(topics, min(3, len(topics)))
    return [f"Quiz {i+1}: {topic}" for i, topic in enumerate(selected_topics)]

def show_chat_interface():
    st.title("Chat with Your Learning Assistant")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Initialize RAG
    rag = init_rag()
    
    if rag is None:
        st.error("Failed to initialize the learning assistant. Please check your configuration.")
        return

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask a question about your textbook"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate assistant response
        with st.chat_message("assistant"):
            response = rag.run(prompt)
            st.markdown(response)
            
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def show_class_list():
    st.title("Deep Learning Educational Tool")
    st.write("Welcome, Professor! Here are your classes:")

    # Add Chat Interface button
    if st.button("Open Chat Interface", key="chat_button"):
        st.session_state.page = "chat"
        st.rerun()

    for class_name in st.session_state.classes:
        if st.button(class_name, key=class_name):
            st.session_state.selected_class = class_name
            st.rerun()

def show_quiz_list():
    st.title(f"Quizzes for {st.session_state.selected_class}")
    
    quizzes = st.session_state.class_quizzes[st.session_state.selected_class]
    
    for quiz in quizzes:
        st.write(quiz)
    
    if st.button("Back to Class List"):
        st.session_state.selected_class = None
        st.rerun()

# Main app
def main():
    st.set_page_config(page_title="DL Educational Tool", page_icon="🎓", layout="wide")

    # Initialize session state
    if 'classes' not in st.session_state:
        st.session_state.classes = generate_fixed_classes()
    
    if 'class_quizzes' not in st.session_state:
        st.session_state.class_quizzes = {class_name: generate_fake_quizzes(class_name) for class_name in st.session_state.classes}
    
    if 'selected_class' not in st.session_state:
        st.session_state.selected_class = None
        
    if 'page' not in st.session_state:
        st.session_state.page = "home"

    # Navigation
    if st.session_state.page == "chat":
        show_chat_interface()
        if st.button("Back to Home"):
            st.session_state.page = "home"
            st.rerun()
    else:
        if st.session_state.selected_class is None:
            show_class_list()
        else:
            show_quiz_list()

if __name__ == "__main__":
    main()