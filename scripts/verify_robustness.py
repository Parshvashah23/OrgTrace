import requests
import json

BASE_URL = "http://127.0.0.1:7860"

def test_robustness():
    print("--- Running Robustness Tests ---")

    # 1. JSON Body (standard)
    print("\nTest 1: JSON Body (task_id)")
    r = requests.post(f"{BASE_URL}/reset", json={"task_id": "decision_archaeology"})
    print(f"Status: {r.status_code}, Task: {r.json().get('task_id') if r.status_code == 200 else r.text}")

    # 2. Query param only
    print("\nTest 2: Query Param (?task_id=)")
    r = requests.post(f"{BASE_URL}/reset?task_id=decision_archaeology")
    print(f"Status: {r.status_code}, Task: {r.json().get('task_id') if r.status_code == 200 else r.text}")

    # 3. Alias 'task' in Query param
    print("\nTest 3: Alias 'task' (?task=)")
    r = requests.post(f"{BASE_URL}/reset?task=decision_archaeology")
    print(f"Status: {r.status_code}, Task: {r.json().get('task_id') if r.status_code == 200 else r.text}")

    # 4. Form data
    print("\nTest 4: Form Data (task_id)")
    r = requests.post(f"{BASE_URL}/reset", data={"task_id": "decision_archaeology"})
    print(f"Status: {r.status_code}, Task: {r.json().get('task_id') if r.status_code == 200 else r.text}")

    # 5. Mixed: Body {} and Query Param ?task_id=...
    print("\nTest 5: Mixed (Empty JSON body + Query Param)")
    r = requests.post(f"{BASE_URL}/reset?task_id=decision_archaeology", json={})
    print(f"Status: {r.status_code}, Task: {r.json().get('task_id') if r.status_code == 200 else r.text}")

    # 6. Step Test (JSON)
    print("\nTest 6: Step (JSON)")
    r = requests.post(f"{BASE_URL}/step", json={
        "action_type": "retrieve_messages",
        "parameters": {"query": "oauth"}
    })
    print(f"Status: {r.status_code}, Action: {r.json().get('reward', {}).get('feedback') if r.status_code == 200 else r.text}")

if __name__ == "__main__":
    test_robustness()
