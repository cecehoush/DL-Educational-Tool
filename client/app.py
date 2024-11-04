import streamlit as st
import random
import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from models.cece import OptimizedEnhancedRAGApplication
except ImportError as e:
    print(f"Error importing OptimizedEnhancedRAGApplication: {e}")
    
@st.cache_resource
def init_rag():
    try:
        file_path = os.path.join(project_root, "textbook.pdf")
        openai_api_key = os.getenv("OPENAI_KEY")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Could not find textbook at {file_path}")
            
        if not openai_api_key:
            raise ValueError("OPENAI_KEY environment variable is not set")
            
        return OptimizedEnhancedRAGApplication(
            file_paths=[file_path],
            openai_api_key=openai_api_key
        )
    except Exception as e:
        st.error(f"Error initializing RAG: {str(e)}")
        return None

def generate_fixed_classes():
    return [
        "CS301: Machine Learning",
        "AI401: Deep Learning",
        "DS201: Computer Vision",
        "CS501: Natural Language Processing"
    ]

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
    
    # Initialize message history in session state if it doesn't exist
    if "message_history" not in st.session_state:
        st.session_state.message_history = []
    
    # Initialize RAG
    rag = init_rag()
    
    if rag is None:
        st.error("Failed to initialize the learning assistant. Please check your configuration.")
        return

    # Display chat messages from history
    for message in st.session_state.message_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input("Ask a question about your textbook"):
        # Clear any previous response for this prompt
        if hasattr(st.session_state, 'last_prompt') and st.session_state.last_prompt == prompt:
            return
            
        st.session_state.last_prompt = prompt
        
        # Add and display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.message_history.append({"role": "user", "content": prompt})
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = rag.run(prompt)
            st.markdown(response)
        st.session_state.message_history.append({"role": "assistant", "content": response})
        
        # Force a rerun to update the UI
        st.rerun()

def show_class_list():
    st.title("Deep Learning Educational Tool")
    st.write("Welcome, Professor! Here are your classes:")

    # Add Chat Interface button
    if st.button("Open Chat Interface", key="chat_button"):
        st.session_state.page = "chat"
        if "message_history" in st.session_state:
            st.session_state.message_history = []
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

def main():
    st.set_page_config(page_title="DL Educational Tool", page_icon="🎓", layout="wide")

    # Initialize session state
    if 'classes' not in st.session_state:
        st.session_state.classes = generate_fixed_classes()
    
    if 'class_quizzes' not in st.session_state:
        st.session_state.class_quizzes = {
            class_name: generate_fake_quizzes(class_name) 
            for class_name in st.session_state.classes
        }
    
    if 'selected_class' not in st.session_state:
        st.session_state.selected_class = None
        
    if 'page' not in st.session_state:
        st.session_state.page = "home"

    # Navigation
    if st.session_state.page == "chat":
        show_chat_interface()
        if st.button("Back to Home"):
            if "message_history" in st.session_state:
                st.session_state.message_history = []
            st.session_state.page = "home"
            st.rerun()
    else:
        if st.session_state.selected_class is None:
            show_class_list()
        else:
            show_quiz_list()

if __name__ == "__main__":
    main()