"""
RouteDesk ML Service — API verification test script.
Tests /health and /predict endpoints for valid and invalid inputs.
"""

from __future__ import annotations

import json
from starlette.testclient import TestClient
from app.main import app

def run_tests():
    print("=" * 60)
    print("RUNNING FASTAPI ML SERVICE TESTS")
    print("=" * 60)

    with TestClient(app) as client:
        # Test 1: GET /health
        print("\n--- Test 1: GET /health ---")
        res = client.get("/health")
        print(f"Status: {res.status_code}")
        data = res.json()
        print(json.dumps(data, indent=2))
        assert res.status_code == 200
        assert data["status"] == "ok"
        assert data["model_loaded"] is True
        assert data["classes_count"] == 10

        # Test 2: POST /predict - Billing inquiry
        print("\n--- Test 2: POST /predict (Billing inquiry) ---")
        req_body = {
            "subject": "Double charged on latest subscription renewal",
            "body": "I was billed twice for my invoice this month on my credit card. Please issue a refund.",
        }
        res = client.post("/predict", json=req_body)
        print(f"Status: {res.status_code}")
        data = res.json()
        print(json.dumps(data, indent=2))
        assert res.status_code == 200
        assert "queue" in data
        assert "confidence" in data
        assert isinstance(data["confidence"], float)
        assert 0.0 <= data["confidence"] <= 1.0

        # Test 3: POST /predict - IT Support inquiry
        print("\n--- Test 3: POST /predict (IT Support) ---")
        req_body = {
            "subject": "VPN connection dropping repeatedly",
            "body": "Every 10 minutes my Cisco AnyConnect client drops connection on office laptop.",
        }
        res = client.post("/predict", json=req_body)
        print(f"Status: {res.status_code}")
        data = res.json()
        print(json.dumps(data, indent=2))
        assert res.status_code == 200
        assert "queue" in data

        # Test 4: POST /predict - Subject only
        print("\n--- Test 4: POST /predict (Subject only) ---")
        req_body = {"subject": "Need password reset link for portal"}
        res = client.post("/predict", json=req_body)
        print(f"Status: {res.status_code}")
        data = res.json()
        print(json.dumps(data, indent=2))
        assert res.status_code == 200
        assert data["queue"] is not None

        # Test 5: POST /predict - Body only
        print("\n--- Test 5: POST /predict (Body only) ---")
        req_body = {"body": "Where do I submit my maternity leave and PTO request documents?"}
        res = client.post("/predict", json=req_body)
        print(f"Status: {res.status_code}")
        data = res.json()
        print(json.dumps(data, indent=2))
        assert res.status_code == 200
        assert data["queue"] is not None

        # Test 6: POST /predict - Invalid input (empty body and subject)
        print("\n--- Test 6: POST /predict (Invalid: empty subject & body) ---")
        req_body = {"subject": "", "body": ""}
        res = client.post("/predict", json=req_body)
        print(f"Status: {res.status_code}")
        data = res.json()
        print(json.dumps(data, indent=2))
        assert res.status_code == 422

        # Test 7: POST /predict - Invalid input (whitespace only)
        print("\n--- Test 7: POST /predict (Invalid: whitespace only) ---")
        req_body = {"subject": "   ", "body": "   "}
        res = client.post("/predict", json=req_body)
        print(f"Status: {res.status_code}")
        data = res.json()
        print(json.dumps(data, indent=2))
        assert res.status_code == 422

        # Test 8: POST /predict - Invalid input (empty object {})
        print("\n--- Test 8: POST /predict (Invalid: empty JSON object) ---")
        req_body = {}
        res = client.post("/predict", json=req_body)
        print(f"Status: {res.status_code}")
        data = res.json()
        print(json.dumps(data, indent=2))
        assert res.status_code == 422

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
