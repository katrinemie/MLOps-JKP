"""
Endpoint tests for the Flask inference API (serve.py).

Disse tests bruger Flasks built-in test-klient og kræver IKKE
at serveren kører — ingen netværksforbindelser bruges.
Dette giver D5.2-bevis: endpointene returnerer korrekte HTTP-statuskoder
og JSON-svar uden at modellen er indlæst (model=None simulerer en kold server).
"""

import sys
import os

# Tilføj src/ til Python-stien så vi kan importere fra serve.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
from serve import app


@pytest.fixture
def client():
    """
    Opretter en Flask test-klient.
    Vi initialiserer IKKE modellen — vi tester kun endpointenes HTTP-adfærd.
    """
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_health_returns_200(client):
    """
    /health skal altid returnere HTTP 200.
    Bruges af load balancers og CI til at verificere serveren kører.
    """
    response = client.get("/health")
    assert response.status_code == 200, (
        f"Forventede 200, fik {response.status_code}"
    )


def test_health_returns_json(client):
    """
    /health skal returnere JSON med 'status'-feltet.
    """
    response = client.get("/health")
    data = response.get_json()
    assert data is not None, "Svar er ikke valid JSON"
    assert "status" in data, f"'status'-felt mangler i: {data}"
    assert data["status"] == "healthy", f"Forventede 'healthy', fik: {data['status']}"


def test_health_has_model_loaded_field(client):
    """
    /health skal inkludere 'model_loaded' feltet
    så CI-pipelinen kan se om modellen er klar.
    """
    response = client.get("/health")
    data = response.get_json()
    assert "model_loaded" in data, f"'model_loaded'-felt mangler i: {data}"


def test_info_returns_200(client):
    """
    /info skal returnere HTTP 200 og metadata om modellen.
    """
    response = client.get("/info")
    assert response.status_code == 200, (
        f"Forventede 200, fik {response.status_code}"
    )


def test_info_returns_classes(client):
    """
    /info skal indeholde 'classes'-feltet med Cat og Dog.
    """
    response = client.get("/info")
    data = response.get_json()
    assert data is not None, "Svar er ikke valid JSON"
    assert "classes" in data, f"'classes'-felt mangler i: {data}"
    assert "Cat" in data["classes"], f"'Cat' mangler i classes: {data['classes']}"
    assert "Dog" in data["classes"], f"'Dog' mangler i classes: {data['classes']}"


def test_predict_without_image_returns_400(client):
    """
    /predict uden billedfil skal returnere HTTP 400.
    Verificerer at API'et validerer input korrekt.
    """
    response = client.post("/predict")
    assert response.status_code == 400, (
        f"Forventede 400 ved manglende billede, fik {response.status_code}"
    )


def test_unknown_endpoint_returns_404(client):
    """
    Ukendte endpoints skal returnere HTTP 404.
    """
    response = client.get("/denne-findes-ikke")
    assert response.status_code == 404, (
        f"Forventede 404, fik {response.status_code}"
    )
