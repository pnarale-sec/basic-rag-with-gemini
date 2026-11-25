
from google import genai
from dotenv import load_dotenv
import os

from embeddings import generate_embedding
from retriever import retrieve_document
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
question = "Can I get my money back after 5 days?"


def main():

    print("Generating document embeddings...")

    # Generate embeddings for all documents
    document_vectors = []

    for document in documents:

        vector = generate_embedding(client, document)

        document_vectors.append(vector)


    print("Document embeddings generated successfully.")


    # Generate question embedding
    question_vector = generate_embedding(client, question)


    # Retrieve the most relevant document
    retrieved_document, similarity_score = retrieve_document(
        question_vector,
        documents,
        document_vectors
    )


    print("\nRetrieved Document:")
    print(retrieved_document)

    print("\nSimilarity Score:")
    print(round(similarity_score, 3))


    # Generate final answer
    answer = generate_answer(
        client,
        question,
        retrieved_document
    )


    if answer:
        print("\nLLM Answer:")
        print(answer)


if __name__ == "__main__":
    main()
