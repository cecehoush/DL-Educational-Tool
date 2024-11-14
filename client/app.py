import streamlit as st
import random
import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from models.llama_RAG_model import OptimizedEnhancedRAGApplication
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
        initial_message = "(Chatbot): "
        if hasattr(st.session_state, 'chat_mode') and st.session_state.chat_mode == "quiz":
            initial_message += "Hello! Please share your quiz question, and I'll help you work through it step by step."
        else:
            initial_message += "Hello, do you have a quiz question you need help with, or do you need general help with Discrete Structures?"
        st.session_state.message_history = [
            {"role": "assistant", "content": initial_message}
        ]
    
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
    if "user_selection" not in st.session_state:
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Quiz question"):
                st.session_state.user_selection = "quiz"
                st.session_state.message_history.append({"role": "assistant", "content": "(Chatbot): Give me your quiz question and we'll solve it step by step."})
                st.rerun()
        with col2:
            if st.button("General question"):
                st.session_state.user_selection = "general"
                st.session_state.message_history.append({"role": "assistant", "content": "(Chatbot): Please enter your general question about Discrete Structures."})
                st.rerun()
    else:
        if st.session_state.user_selection == "quiz":
            prompt_text = "Please enter your quiz question:"
        else:
            prompt_text = "Please enter your general question about Discrete Structures:"

        if prompt := st.chat_input(prompt_text):
            # Clear any previous response for this prompt
            if hasattr(st.session_state, 'last_prompt') and st.session_state.last_prompt == prompt:
                return
                
            st.session_state.last_prompt = prompt
            
            # Add and display user message
            with st.chat_message("user"):
                st.markdown(f"(Student): {prompt}")
            st.session_state.message_history.append({"role": "user", "content": f"(Student): {prompt}"})
            
            # Generate and display assistant response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = rag.run(prompt)
                st.markdown(response)
            st.session_state.message_history.append({"role": "assistant", "content": response})
            
            # Force a rerun to update the UI
            st.rerun()

def show_student_course_options():
    st.title(f"Learning Assistant - {st.session_state.selected_course}")
    
    st.markdown("<h3 style='text-align: center;'>What would you like to discuss today?</h3>", unsafe_allow_html=True)
    
    st.write("")
    
    col1, col2, col3 = st.columns([2, 2, 2]) 

    with col2:
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("General Course Questions", use_container_width=True):
                st.session_state.page = "chat"
                st.session_state.chat_mode = "general"
                st.rerun()
        with col_b:
            if st.button("Quiz Help", use_container_width=True):
                st.session_state.page = "chat"
                st.session_state.chat_mode = "quiz"
                st.rerun()
    
    st.write("")  
    

    col1, col2, col3 = st.columns([3, 1, 3]) 
    with col2:
        if st.button("Back to Courses"):
            st.session_state.selected_course = None
            st.rerun()

def show_student_view():
    st.title("Student Learning Assistant")
    
    st.write("Your Courses:")
    courses = ["Discrete Structures", "CS101: Intro to Programming", "MATH201: Linear Algebra"]
    for course in courses:
        if st.button(course, key=course, use_container_width=False):
            st.session_state.selected_course = course
            st.rerun()
    
    st.write("")
    st.markdown("---")  
    st.write("Additional Features:")
    
    st.markdown("""
        <style>
        .feature-button {
            background-color: #f0f2f6;
            border: 1px solid #e0e0e0;
            padding: 0.5rem;
            border-radius: 4px;
            margin: 0.25rem 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])  
    with col2:
        if st.button("Previous Conversations", use_container_width=True, key="prev_conv"):
            st.info("Feature coming soon!")
            
        if st.button("Practice Mode", use_container_width=True, key="practice"):
            st.info("Feature coming soon!")

def show_class_list():
    st.title("Deep Learning Educational Tool")
    st.write("Welcome, Professor! Here are your classes:")

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
    
    # Initializing session states :v
    if 'role' not in st.session_state:
        st.session_state.role = None
    if 'page' not in st.session_state:
        st.session_state.page = "home"
    if 'classes' not in st.session_state:
        st.session_state.classes = generate_fixed_classes()
    
    if 'class_quizzes' not in st.session_state:
        st.session_state.class_quizzes = {
            class_name: generate_fake_quizzes(class_name) 
            for class_name in st.session_state.classes
        }
    
    if 'selected_class' not in st.session_state:
        st.session_state.selected_class = None
    if 'selected_course' not in st.session_state:
        st.session_state.selected_course = None
    
    if st.session_state.role is None:
        st.title("Welcome to the Learning Assistant")
        st.write("")
        st.write("")
        _, col2, _ = st.columns([1, 2, 1])
        with col2:
            if st.button("I'm a Student", use_container_width=True):
                st.session_state.role = "student"
                st.rerun()
            st.write("")
            if st.button("I'm a Professor", use_container_width=True):
                st.session_state.role = "professor"
                st.rerun()
        return

    # Navigation :v
    if st.session_state.page == "chat":
        show_chat_interface()
        if st.button("Back"):
            st.session_state.page = "home"
            if "message_history" in st.session_state:
                st.session_state.message_history = []
            st.rerun()
    else:
        if st.session_state.role == "student":
            if st.session_state.selected_course is None:
                show_student_view()
            else:
                show_student_course_options()
        else:
            if st.session_state.selected_class is None:
                show_class_list()
            else:
                show_quiz_list()

if __name__ == "__main__":
    main()