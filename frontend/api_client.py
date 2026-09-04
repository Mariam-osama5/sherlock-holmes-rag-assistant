import os

import requests
from dotenv import load_dotenv


load_dotenv()


BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://127.0.0.1:8000"
)


def ask_question(question: str):
    response = requests.post(
        f"{BACKEND_URL}/query",
        json={"question": question},
        timeout=120
    )

    response.raise_for_status()

    return response.json()