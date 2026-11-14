# Run with: pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def test_health():
    response = client.get('/api/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'

def test_dashboard():
    response = client.get('/api/dashboard')
    assert response.status_code == 200
    assert response.json()['rows'] > 0
