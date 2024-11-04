from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from functools import lru_cache
import logging
import PyPDF2
import os
import dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_documents(file_paths):
    """Load documents from files (txt or pdf) and convert them to Document objects."""
    docs = []
    for file_path in file_paths:
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == '.pdf':
                # Handle PDF files
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                    if text.strip():
                        docs.append(Document(page_content=text))
                        logger.info(f"Successfully loaded PDF: {file_path}")
                    else:
                        logger.warning(f"Empty PDF found: {file_path}")
            
            elif file_extension == '.txt':
                # Handle text files
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    if text.strip():
                        docs.append(Document(page_content=text))
                        logger.info(f"Successfully loaded text file: {file_path}")
                    else:
                        logger.warning(f"Empty text file found: {file_path}")
            else:
                logger.warning(f"Unsupported file type: {file_path}")
                
        except Exception as e:
            logger.error(f"Error loading document {file_path}: {str(e)}")
    return docs

def create_text_chunks(documents, chunk_size=500, chunk_overlap=50):
    """Split documents into smaller chunks with increased size and overlap."""
    if not documents:
        raise ValueError("No documents provided for splitting")

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    try:
        doc_splits = text_splitter.split_documents(documents)
        logger.info(f"Created {len(doc_splits)} text chunks")
        return doc_splits
    except Exception as e:
        logger.error(f"Error splitting documents: {str(e)}")
        raise

def create_vectorstore(doc_splits, openai_api_key):
    """Create a vector store from document chunks."""
    try:
        vectorstore = SKLearnVectorStore.from_documents(
            documents=doc_splits,
            embedding=OpenAIEmbeddings(openai_api_key=openai_api_key),
        )
        logger.info("Vector store created successfully")
        return vectorstore
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        raise

@lru_cache(maxsize=1)
def load_documents_cached(file_path):
    """Cache document loading to avoid repeated disk reads.
    Now accepts a single file path instead of a tuple."""
    try:
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                if text.strip():
                    return [Document(page_content=text)]
        
        elif file_extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                if text.strip():
                    return [Document(page_content=text)]
        
        return []
    except Exception as e:
        logger.error(f"Error loading document {file_path}: {str(e)}")
        return []

@lru_cache(maxsize=1)
def create_vectorstore_cached(doc_splits_hash, openai_api_key):
    """Cache vectorstore creation to avoid recomputing embeddings."""
    return create_vectorstore(doc_splits_hash[1], openai_api_key)

# Optimized Enhanced RAG Application
class OptimizedEnhancedRAGApplication:
    def __init__(self, file_paths, openai_api_key, model_name="llama3.1"):
        # Load documents one by one using cached function
        self.docs = []
        for file_path in file_paths:
            self.docs.extend(load_documents_cached(file_path))
        
        # Create text chunks
        self.doc_splits = create_text_chunks(self.docs, chunk_size=500, chunk_overlap=100)
        
        # Initialize vectorstore (without caching for now to fix the error)
        self.vectorstore = create_vectorstore(self.doc_splits, openai_api_key)
        
        # Optimize retrieval parameters
        self.k = min(3, len(self.doc_splits))
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={
                "k": self.k,
                "fetch_k": self.k * 2
            }
        )
        
        # Initialize conversation history with fixed size
        self.conversation_history = []
        self.max_history_length = 24
        
        # Initialize LLM with optimized settings
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.4,
        )

        # Optimized prompt template
        self.prompt = PromptTemplate(
            template="""You are a helpful teaching assistant for a computer science course.
Engage naturally with students while maintaining a focus on their learning journey.
DO NOT directly give answers.
Provide step-by-step guidance, but only one step per response.

---

Context from textbook (for your reference only; use only if directly relevant):
{documents}

---

Previous conversation (for your reference only; do not repeat or mention this):
{conversation_history}

---

Student's Question/Input:
{question}

---

Instructions:

1. **Understand and Focus on the Student's Current Input:**
   - Carefully read the student's latest message.
   - Determine if they have provided an answer, asked a question, or need clarification.

2. **Acknowledge and Confirm Correct Answers:**
   - If the student has arrived at the correct final answer:
     - Clearly confirm the correctness with positive reinforcement.
     - Briefly summarize the solution and its relevance to the original question.
     - Offer to assist with any further questions or ask if they have another quiz question.
   - If the student provides a correct intermediate step:
     - Confirm it's correct.
     - Guide them to the next step.

3. **Gently Correct Incorrect Answers:**
   - If the student provides an incorrect answer:
     - Gently inform them it's not correct.
     - Explain the mistake without negativity.
     - Guide them toward the correct solution with hints.

4. **Avoid Unnecessary Repetition:**
   - Do not repeat the same information or phrases.
   - Avoid rehashing steps that the student has already completed correctly.

5. **Provide Clear and Direct Guidance:**
   - Keep responses concise and directly related to the student's input.
   - Offer one clear step at a time to advance the problem-solving process.

6. **Recognize Completion and Progress the Conversation:**
   - When the problem is solved, acknowledge it.
   - Ask if the student has any other questions or needs further assistance.

7. **Maintain a Supportive Tone:**
   - Use encouraging language.
   - Foster a positive and efficient learning environment.

8. **Stay Relevant and Focused:**
   - Use textbook content only if directly relevant.
   - Avoid introducing unrelated topics.

9. **Do Not Give Direct Answers:**
   - Encourage the student to arrive at the answer themselves.
   - Use guiding questions to lead them.

---

Your response:
"""

,
            input_variables=["question", "documents", "conversation_history"],
        )

        self.rag_chain = self.prompt | self.llm | StrOutputParser()
        logger.info("Optimized RAG application initialized")

    def run(self, question):
        """Optimized RAG pipeline execution."""
        try:
            # Retrieve relevant documents
            documents = self.retriever.invoke(question)
            
            # Process only the most relevant parts of the documents
            doc_texts = "\n\n".join([
                doc.page_content[:500] for doc in documents
            ])
            
            # Format conversation history but don't include current question
            conversation_context = "\n".join([
                f"{'Student' if i % 2 == 0 else 'Assistant'}: {msg}"
                for i, msg in enumerate(self.conversation_history[-self.max_history_length:])
            ])
            
            # Get the answer
            answer = self.rag_chain.invoke({
                "question": question,
                "documents": doc_texts,
                "conversation_history": conversation_context
            })
            
            # Update conversation history
            self.conversation_history.extend([question, answer])
            if len(self.conversation_history) > self.max_history_length:
                self.conversation_history = self.conversation_history[-self.max_history_length:]
            
            return f"(Chatbot): {answer}"

        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return "(Chatbot): I apologize, but I encountered an error while processing your question."

def main():
    # Configuration
    file_paths = ["textbook.pdf"]
    openai_api_key = dotenv.get_key(".env", "OPENAI_KEY")
    
    try:
        rag_application = OptimizedEnhancedRAGApplication(
            file_paths=file_paths,
            openai_api_key=openai_api_key
        )

        # Example usage
        questions = [
            "What is a set in mathematics?",
            "Can you explain that in a simpler way?",
            "Can you give me a real-world example?"
        ]
        
        for question in questions:
            print(f"\n(Student): {question}")
            answer = rag_application.run(question)
            print(f"Assistant: {answer}\n")
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        print("An error occurred while running the application.")

if __name__ == "__main__":
    main()