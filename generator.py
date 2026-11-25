
from google.genai import errors

GENERATION_MODEL = "gemini-3.8-flash"


def generate_answer(client, question, context):
    """
    Generate an answer using the retrieved context.
    """

    prompt = f"""
You are a helpful assistant.

Answer the user's question using ONLY the provided context.

If the answer is not available in the context,
say that you don't have enough information.

Context:
{context}

Question:
{question}

Answer:
"""

    try:

        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt
        )

        return response.text

    except errors.ServerError as e:

        print("Gemini server is temporarily unavailable.")
        print("Error:", e)

        return None

    except Exception as e:

        print("Something went wrong:", e)

        return None
