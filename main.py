
from google import genai
from dotenv import load_dotenv
import os

from embeddings import generate_embedding
from retriever import retrieve_documents
from generator import generate_answer


# Load environment variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. Add it to your .env file."
    )


# Initialize Gemini client
client = genai.Client(api_key=api_key)


# Sample knowledge base
documents = [
    "Python classes happen monday to friday",
    "Students get 3 interview opportunities",
    "Refund requests are allowed within 7 days",
]


# User question
question = "when python classes happens"


def main():

    print("Generating document embeddings...")

    document_vectors = []

    for document in documents:
        vector = generate_embedding(client, document)
        document_vectors.append(vector)

    print("Document embeddings generated successfully.")

    # Generate question embedding
    question_vector = generate_embedding(client, question)

    # Retrieve top 3 relevant documents
    retrieved_documents = retrieve_documents(
        question_vector,
        documents,
        document_vectors,
        top_k=3
    )

    # Display retrieved documents
    print("\nRetrieved Documents:")

    if not retrieved_documents:
        print("No relevant documents found.")
        return

    for item in retrieved_documents:
        print("\nDocument:", item["document"])
        print("Similarity:", item["score"])

    # Combine retrieved documents into context
    context = "\n".join(
        item["document"] for item in retrieved_documents
    )

    # Generate final answer
    answer = generate_answer(
        client,
        question,
        context
    )

    if answer:
        print("\nLLM Answer:")
        print(answer)


if __name__ == "__main__":
    main()
