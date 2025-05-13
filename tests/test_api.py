import pytest
import httpx
from fastapi.testclient import TestClient
from src.api.app import app
from typing import Dict, Any

class TestSpamClassifierAPI:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_health_check(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

    def test_predict_spam_email(self, client):
        test_email = {
            "subject": "Urgent: Win Millions Now!",
            "body": "Congratulations! You have won a free iPhone. Click here to claim...",
            "sender": "noreply@suspicious.com"
        }
        
        response = client.post("/predict", json=test_email)
        assert response.status_code == 200
        
        result = response.json()
        assert "prediction" in result
        assert "probability" in result
        assert result["prediction"] in ["spam", "not_spam"]
        assert 0 <= result["probability"] <= 1

    def test_predict_invalid_input(self, client):
        invalid_email = {
            "subject": "",
            "body": "",
            "sender": ""
        }
        
        response = client.post("/predict", json=invalid_email)
        assert response.status_code == 422

    def test_bulk_prediction(self, client):
        test_emails = [
            {
                "subject": "Urgent: Win Millions Now!",
                "body": "Congratulations! You have won a free iPhone.",
                "sender": "noreply@suspicious.com"
            },
            {
                "subject": "Project Meeting Tomorrow",
                "body": "Hi team, we'll discuss the quarterly report.",
                "sender": "manager@company.com"
            }
        ]
        
        response = client.post("/predict/bulk", json=test_emails)
        assert response.status_code == 200
        
        results = response.json()
        assert len(results) == len(test_emails)
        
        for result in results:
            assert "prediction" in result
            assert "probability" in result
            assert result["prediction"] in ["spam", "not_spam"]
            assert 0 <= result["probability"] <= 1

    def test_model_performance_metrics(self, client):
        response = client.get("/metrics")
        assert response.status_code == 200
        
        metrics = response.json()
        expected_metrics = [
            "precision", 
            "recall", 
            "f1_score", 
            "auc_roc"
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
            assert 0 <= metrics[metric] <= 1