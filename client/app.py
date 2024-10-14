import streamlit as st
import random

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
    selected_topics = random.sample(topics, min(3, len(topics)))  # Ensure unique topics
    return [f"Quiz {i+1}: {topic}" for i, topic in enumerate(selected_topics)]

# Main app
def main():
    st.set_page_config(page_title="DL Educational Tool", page_icon="🎓", layout="wide")

    if 'classes' not in st.session_state:
        st.session_state.classes = generate_fixed_classes()
    
    if 'class_quizzes' not in st.session_state:
        st.session_state.class_quizzes = {class_name: generate_fake_quizzes(class_name) for class_name in st.session_state.classes}
    
    if 'selected_class' not in st.session_state:
        st.session_state.selected_class = None

    if st.session_state.selected_class is None:
        show_class_list()
    else:
        show_quiz_list()

def show_class_list():
    st.title("Deep Learning Educational Tool")
    st.write("Welcome, Professor! Here are your classes:")

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

if __name__ == "__main__":
    main()