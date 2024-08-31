# RAG GAN Research Paper Q&A

This application uses Retrieval-Augmented Generation (RAG) to answer questions about Generative Adversarial Networks (GANs) based on a collection of 100+ research papers.

## Dataset

[Link to the GAN research papers dataset](https://drive.google.com/drive/folders/10zPpoPCi-NSeSbfue-yUhTS-pyKQNizW?usp=sharing)

## Description

This Streamlit application leverages the power of Large Language Models (LLMs) and vector databases to provide accurate answers to questions about GANs. It uses the Groq API for language modeling and FAISS for efficient similarity search among document embeddings.

Key features:
- Uses Groq's Gemma2-9b-It model for question answering
- Employs HuggingFace's BGE embeddings for document vectorization
- Utilizes FAISS for fast similarity search
- Provides response times and similarity search results for transparency

## Setup Instructions

1. Clone the repository:
   ```
   git clone https://github.com/NastyRunner13/rag-gan-qa.git
   cd rag-gan-qa
   ```

2. Install required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the root directory with the following contents:
   ```
   HF_TOKEN=your_huggingface_token
   LANGCHAIN_API_KEY=your_langchain_api_key
   GROQ_API_KEY=your_groq_api_key
   ```

4. Prepare your data:
   Place your GAN research papers (in PDF format) in a folder named `ResearchPapers` in the root directory of the project.

5. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

## Usage

1. Open the application in your web browser (typically at `http://localhost:8501`).
2. Click the "Create Document Embeddings" button to process the research papers and create the vector database.
3. Once the vector database is ready, enter your question about GANs in the text input field.
4. The application will display the answer, along with the response time and relevant document excerpts.

## Notes

- The first run may take some time as it needs to process all the research papers and create embeddings.
- Ensure you have sufficient disk space and memory to handle the vector database creation and storage.

## Contributing

Contributions, issues, and feature requests are welcome!
