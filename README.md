# Basic RAG Implementation Using Gemini

A beginner-friendly **Retrieval-Augmented Generation (RAG)** implementation using Python and Google's Gemini API. This project demonstrates how text embeddings and cosine similarity can retrieve relevant information before an LLM generates a context-based response.

## 🚀 Project Overview

This project implements a basic RAG pipeline that retrieves the most relevant document from a small knowledge base based on a user's question.

Instead of directly sending a question to an LLM, the application:

1. Converts documents into numerical embeddings.
2. Converts the user's question into an embedding.
3. Calculates cosine similarity between the question and documents.
4. Retrieves the most relevant document.
5. Sends the retrieved context to Gemini.
6. Generates an answer using the provided context.

## 🧠 Concepts Learned

- Large Language Models (LLMs)
- Generative AI
- Text Embeddings
- Semantic Search
- Cosine Similarity
- Vector Similarity
- Document Retrieval
- Retrieval-Augmented Generation (RAG)
- Prompt Engineering
- Python Modular Programming

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Google GenAI SDK | Gemini API integration |
| Gemini Embedding Model | Text embedding generation |
| Gemini Flash | Answer generation |
| NumPy | Vector operations and cosine similarity |
| python-dotenv | Environment variable management |

## 📁 Project Structure

```text
basic-rag-with-gemini/
│
├── main.py
├── embeddings.py
├── retriever.py
├── generator.py
├── requirements.txt
├── README.md
├── .env.example
└── .gitignore
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/pnarale-sec/basic-rag-with-gemini.git
```

### 2. Navigate to the project directory

```bash
cd basic-rag-with-gemini
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

**Windows PowerShell:**

```powershell
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Configure your Gemini API key

Create a `.env` file in the project root directory.

Add your API key:

```env
GEMINI_API_KEY=your_api_key_here
```

Get your API key from [Google AI Studio](https://aistudio.google.com/apikey).

**Note:** Never upload your actual `.env` file or API key to GitHub.

### 7. Run the application

```bash
python main.py
```

## 💡 Example

**Sample question:**

Can I get my money back after 5 days?

**Relevant document retrieved:**

Refund requests are allowed within 7 days.

The retrieved document is passed to Gemini to generate a context-based answer.

## 📌 Current Limitations

- Uses a small, predefined collection of documents.
- Generates document embeddings every time the application runs.
- Retrieves only the highest-scoring document.
- Does not use a persistent vector database.
- Does not currently implement a similarity threshold.

## 🔮 Future Improvements

- Implement Top-K document retrieval.
- Add similarity threshold filtering.
- Store embeddings for reuse.
- Integrate a vector database such as ChromaDB or FAISS.
- Support PDF document ingestion.
- Build a RAG-based chatbot using FastAPI.

## 🎯 Learning Objective

This project was developed as a hands-on learning exercise to understand the fundamentals of embeddings, semantic search, document retrieval, and Retrieval-Augmented Generation using Python and Gemini.

---

**Author:** Prathamesh Narale

**GitHub:** [pnarale-sec](https://github.com/pnarale-sec)
