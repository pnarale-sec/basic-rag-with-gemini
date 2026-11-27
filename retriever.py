import numpy as np


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return np.dot(a, b) / denominator



def retrieve_documents(
    question_vector,
    documents,
    document_vectors,
    top_k=3,
    threshold=0.60
):
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")

    if len(documents) != len(document_vectors):
        raise ValueError("Documents and embeddings count must match")

    if not documents:
        return []

    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")

    if len(question_vector) == 0:
        raise ValueError("Question embedding cannot be empty")
    
    scores = []
    for vector in document_vectors:
        score = cosine_similarity(question_vector, vector)
        scores.append(score)

    # Sort indexes from highest score to lowest
    sorted_indexes = np.argsort(scores)[::-1]

    # Select top K indexes
    top_k_indexes = sorted_indexes[:top_k]

    # Store relevant documents
    retrieved_documents = []

    for index in top_k_indexes:

        if scores[index] >= threshold:

            retrieved_documents.append({
                "document": documents[index],
                "score": scores[index]
            })

    # Return matching documents
    return retrieved_documents
