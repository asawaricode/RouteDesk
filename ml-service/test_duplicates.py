"""
Test script for semantic duplicate detection in the RouteDesk ML service.
Tests:
1. Empty ticket candidates (empty history)
2. Highly similar tickets (semantic match)
3. Unrelated tickets (low similarity)
"""

from fastapi.testclient import TestClient
from app.main import app

def run_tests():
    with TestClient(app) as client:
        # Test 1: Health check
        res = client.get("/health")
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "ok"
        assert data["embedding_model_loaded"] is True
        assert data["embedding_model"] == "all-MiniLM-L6-v2"
        print("[PASS] /health reports embedding model loaded")

        # Test 2: Empty candidates (empty history)
        payload = {
            "subject": "VPN client connection dropping",
            "body": "Gateway unreachable error when connecting to corporate VPN",
            "candidates": [],
            "threshold": 0.55
        }
        res = client.post("/duplicates", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["has_duplicates"] is False
        assert data["candidate_count"] == 0
        assert len(data["duplicates"]) == 0
        print("[PASS] Empty candidates handled correctly: has_duplicates = False")

        # Test 3: Similar ticket match
        payload = {
            "subject": "Forgot account password",
            "body": "I am unable to sign in to my user portal because my password reset link is invalid.",
            "candidates": [
                {
                    "id": 1,
                    "subject": "Unable to reset password",
                    "body": "I cannot log into my account after requesting a password reset.",
                    "predicted_queue": "Billing and Payments"
                },
                {
                    "id": 2,
                    "subject": "Refund request for double billing",
                    "body": "My Visa card was charged two times for invoice #1002.",
                    "predicted_queue": "Billing and Payments"
                }
            ],
            "threshold": 0.55,
            "top_k": 3
        }
        res = client.post("/duplicates", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["has_duplicates"] is True
        assert len(data["duplicates"]) == 2

        # Top match should be ticket #1 with high similarity (> 0.70)
        top_match = data["duplicates"][0]
        assert top_match["id"] == 1
        assert top_match["similarity_score"] >= 0.70, f"Expected >= 0.70, got {top_match['similarity_score']}"

        # Second match should be ticket #2 with lower similarity (< 0.35)
        second_match = data["duplicates"][1]
        assert second_match["id"] == 2
        assert second_match["similarity_score"] < 0.35, f"Expected < 0.35, got {second_match['similarity_score']}"
        print(f"[PASS] Similar ticket detected correctly: Top match score = {top_match['similarity_score']}")

        # Test 4: Unrelated ticket
        payload = {
            "subject": "Inquiring about company holiday schedule and vacation days",
            "body": "Where can employees find the official PTO calendar for the upcoming quarter?",
            "candidates": [
                {
                    "id": 1,
                    "subject": "Unable to reset password",
                    "body": "I cannot log into my account after requesting a password reset.",
                    "predicted_queue": "Billing and Payments"
                }
            ],
            "threshold": 0.55
        }
        res = client.post("/duplicates", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["has_duplicates"] is False
        top_match = data["duplicates"][0]
        assert top_match["similarity_score"] < 0.40, f"Expected < 0.40, got {top_match['similarity_score']}"
        print(f"[PASS] Unrelated ticket passed correctly: has_duplicates = False (Score: {top_match['similarity_score']})")

if __name__ == "__main__":
    run_tests()
    print("\n[ALL TESTS PASSED] Semantic duplicate detection test suite succeeded.")
