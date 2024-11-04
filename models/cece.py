from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
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

class RAGApplication:
    def __init__(self, file_paths, openai_api_key, model_name="llama3.1"):
        # Load and process documents
        self.docs = load_documents(file_paths)
        self.doc_splits = create_text_chunks(self.docs)

        # Calculate maximum possible k value based on number of chunks
        self.max_k = len(self.doc_splits)
        logger.info(f"Maximum possible k value: {self.max_k}")

        # Create vector store with appropriate k value
        self.vectorstore = create_vectorstore(self.doc_splits, openai_api_key)
        # Set k to be the minimum of 4 and the number of available chunks
        self.k = min(4, self.max_k)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.k})

        # Initialize LLM
        self.llm = ChatOllama(
            model=model_name,
            temperature=0,
        )

        # Create prompt template
        self.prompt = PromptTemplate(
            template="""You are a knowledgeable assistant helping students understand concepts from their textbook. 
            Use the following textbook excerpts to answer the question.
            If the answer is not contained in the excerpts, say "I cannot find information about this in the provided text."
            Use three sentences maximum and keep the answer concise and focused.

            Textbook excerpts:
            {documents}

            Question: {question}
            Answer:""",
            input_variables=["question", "documents"],
        )

        # Create chain
        self.rag_chain = self.prompt | self.llm | StrOutputParser()

        logger.info(f"RAG application initialized successfully with k={self.k}")

    def run(self, question):
        """Run the RAG pipeline on a question."""
        try:
            # Log the retrieval attempt
            logger.info(f"Attempting to retrieve {self.k} documents for the question")

            # Retrieve relevant documents
            documents = self.retriever.invoke(question)
            logger.info(f"Successfully retrieved {len(documents)} documents")

            # Extract content from retrieved documents
            doc_texts = "\n\n".join([doc.page_content for doc in documents])
            
            # Get the answer from the language model
            answer = self.rag_chain.invoke({
                "question": question,
                "documents": doc_texts
            })

            return answer

        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return "I apologize, but I encountered an error while processing your question."

class EnhancedRAGApplication:
    def __init__(self, file_paths, openai_api_key, model_name="llama3.1"):
        # Load and process documents
        self.docs = load_documents(file_paths)
        self.doc_splits = create_text_chunks(self.docs)

        # Calculate maximum possible k value based on number of chunks
        self.max_k = len(self.doc_splits)
        logger.info(f"Maximum possible k value: {self.max_k}")

        # Create vector store with appropriate k value
        self.vectorstore = create_vectorstore(self.doc_splits, openai_api_key)
        self.k = min(4, self.max_k)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.k})
        
        # Initialize conversation history
        self.conversation_history = []
        
        # Initialize LLM with slightly higher temperature for more natural responses
        self.llm = ChatOllama(
            model=model_name,
            temperature=0.2,
        )

        # Enhanced prompt template
        self.prompt = PromptTemplate(
            template="""You are an intelligent and supportive teaching assistant helping students understand concepts. 
            Your goal is to explain concepts clearly while connecting them to the student's current understanding.

            Here are relevant excerpts from the course textbook:
            {documents}

            Previous conversation context:
            {conversation_history}

            Current question: {question}

            Instructions for responding:
            1. First, ground your response in the textbook content when available.
            2. Then, expand on the concept using general knowledge and examples if needed.
            3. Make connections to related concepts and real-world applications.
            4. If the student seems confused, try explaining the concept in a different way.
            5. If the textbook doesn't contain relevant information, provide a helpful explanation while noting that it's supplementary to the course material.

            Use a natural, conversational tone while maintaining accuracy. Encourage deeper understanding through thoughtful explanation.

            Response:""",
            input_variables=["question", "documents", "conversation_history"],
        )

        self.rag_chain = self.prompt | self.llm | StrOutputParser()
        logger.info("Enhanced RAG application initialized successfully")

    def run(self, question):
        """Run the enhanced RAG pipeline on a question."""
        try:
            # Retrieve relevant documents
            documents = self.retriever.invoke(question)
            doc_texts = "\n\n".join([doc.page_content for doc in documents])
            
            # Format conversation history
            conversation_context = "\n".join([
                f"{'Student' if i % 2 == 0 else 'Assistant'}: {msg}"
                for i, msg in enumerate(self.conversation_history[-4:])  # Keep last 4 exchanges
            ])
            
            # Get the answer
            answer = self.rag_chain.invoke({
                "question": question,
                "documents": doc_texts,
                "conversation_history": conversation_context
            })
            
            # Update conversation history
            self.conversation_history.extend([question, answer])
            
            return answer

        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return "I apologize, but I encountered an error while processing your question."

def main():
    # Configuration
    file_paths = ["textbook.pdf"]
    openai_api_key = dotenv.get_key(".env", "OPENAI_KEY")
    
    try:
        # Initialize the Enhanced RAG application
        rag_application = EnhancedRAGApplication(
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
            print("\nStudent:", question)
            answer = rag_application.run(question)
            print("Assistant:", answer, "\n")
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        print("An error occurred while running the application.")

if __name__ == "__main__":
    main()