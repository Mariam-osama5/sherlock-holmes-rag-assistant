from pathlib import Path
import os

import requests
from dotenv import load_dotenv


load_dotenv(Path(__file__).with_name(".env"))


API_BASE_URL = os.getenv("API_BASE_URL")

if not API_BASE_URL:
    raise RuntimeError(
        "API_BASE_URL is not set. Create frontend/.env from .env.example."
    )


def ask_question(question: str):
    response = requests.post(
        f"{API_BASE_URL}/query",
        json={"question": question},
        timeout=120,
    )

    response.raise_for_status()

    return response.json()