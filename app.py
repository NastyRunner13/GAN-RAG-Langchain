import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check and set environment variables
required_env_vars = ["HF_TOKEN", "LANGCHAIN_API_KEY", "GROQ_API_KEY"]
for var in required_env_vars:
    if not os.getenv(var):
        st.error(f"Missing environment variable: {var}")
        st.stop()

os.environ['LANGCHAIN_PROJECT'] = "RAG DOCUMENT Q&A Groq"
os.environ['LANGCHAIN_TRACING_V2'] = "true"

# Initialize LLM
@st.cache_resource
def init_llm():
    return ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="Gemma2-9b-It")

llm = init_llm()

# Define prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    </context>
    Question:{input}
    """
)

# Initialize embeddings
@st.cache_resource
def init_embeddings():
    return HuggingFaceBgeEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = init_embeddings()

# Create or load vector store
@st.cache_data
def create_vector_store():
    try:
        loader = PyPDFDirectoryLoader("./ResearchPapers")  # Relative path
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs)
        return FAISS.from_documents(final_documents, embeddings)
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

# Streamlit UI
st.title("RAG Document Q&A with Groq and LLaMA 3")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if st.button("Create Document Embeddings"):
    with st.spinner("Creating vector database..."):
        st.session_state.vector_store = create_vector_store()
    if st.session_state.vector_store:
        st.success("Vector database is ready")

user_prompt = st.text_input("Enter your query from the research papers")

if user_prompt and st.session_state.vector_store:
    try:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vector_store.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': user_prompt})
        process_time = time.process_time() - start

        st.write(response['answer'])
        st.info(f"Response Time: {process_time:.2f} seconds")

        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write("--------------")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
elif user_prompt:
    st.warning("Please create document embeddings first.")