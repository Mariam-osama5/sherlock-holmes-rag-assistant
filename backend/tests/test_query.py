from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_query_happy_path(monkeypatch):
    def fake_retrieve_documents(question, top_k=5):
        return [
            {
                "text": "Sherlock Holmes is a detective.",
                "page_number": 123,
                "chunk_index": 0,
                "source": "The Complete Sherlock Holmes",
            }
        ]

    def fake_generate_answer(question, retrieved_chunks):
        return "Sherlock Holmes is a detective. Source: Page 123"

    monkeypatch.setattr(
        "app.api.routes.query.retrieve_documents",
        fake_retrieve_documents
    )

    monkeypatch.setattr(
        "app.api.routes.query.generate_answer",
        fake_generate_answer
    )

    response = client.post(
        "/query",
        json={"question": "Who is Sherlock Holmes?"}
    )

    assert response.status_code == 200

    data = response.json()

    assert "answer" in data
    assert "sources" in data
    assert data["sources"] == ["Page 123"]


def test_query_invalid_input():
    response = client.post(
        "/query",
        json={}
    )

    assert response.status_code == 422