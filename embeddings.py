
EMBEDDING_MODEL = "gemini-embedding-2"


def generate_embedding(client, text):
    """
    Generate an embedding vector for the given text.
    """

    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text
    )

    vector = result.embeddings[0].values

    return vector
