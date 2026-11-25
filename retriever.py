
import numpy as np


def cosine_similarity(a, b):
    """
    Calculate cosine similarity between two vectors.
    """

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    similarity = np.dot(a, b) / (norm_a * norm_b)

    return float(similarity)


def retrieve_document(question_vector, documents, document_vectors):
    """
    Find the most relevant document using cosine similarity.
    """

    scores = []

    print("\nSimilarity Scores:")

    for document, vector in zip(documents, document_vectors):

        score = cosine_similarity(question_vector, vector)

        scores.append(score)

        print(f"{score:.3f} - {document}")

    best_index = int(np.argmax(scores))

    retrieved_document = documents[best_index]
    best_score = scores[best_index]

    return retrieved_document, best_score
