import requests
import time

BASE = "https://Parshva06-orgtrace.hf.space"

def check():
    print(f"🔍 Checking Space: {BASE}")
    
    # 1. Health check (Root)
    try:
        r = requests.get(BASE + "/")
        if r.status_code == 200:
            print("✅ Root Health OK:", r.json())
        else:
            print(f"❌ Root Health failed: {r.status_code}")
    except Exception as e:
        print(f"❌ Root Health request failed: {e}")

    # 2. OpenEnv tasks check
    try:
        r = requests.get(BASE + "/tasks")
        if r.status_code == 200:
            print("✅ /tasks OK:", [t['id'] for t in r.json()])
        else:
            print(f"❌ /tasks failed: {r.status_code}")
    except Exception as e:
        print(f"❌ /tasks request failed: {e}")

    # 3. reset() ping (requested by tutorial)
    print("⏳ Pinging reset() - may take a moment to wake Space...")
    try:
        r = requests.post(BASE + "/reset", params={"task_id": "decision_archaeology"})
        if r.status_code == 200:
            print("✅ reset() OK")
            obs = r.json().get("observation", {})
            print("   Observation received (ID):", obs.get("task_id"))
        else:
            print(f"❌ reset() failed: {r.status_code}")
            print(r.text)
    except Exception as e:
        print(f"❌ reset() request failed: {e}")

if __name__ == "__main__":
    check()
