import requests
import time
import sys

def test_api():
    url = "http://localhost:8000/predict"

    # Dummy data representing a likely Dengue case (High temp, headache, joint pain)
    payload = {
        "Age": 30,
        "Sex": 1,
        "temperature": 39.5,
        "headache": 1,
        "joint_pain": 1,
        "rash": 1,
        "vomiting": 0,
        "fatigue": 1,
        "chills": 1,
        "fever_pattern": 2,
        "travel_to_hot_area": 1,
        "mosquito_exposure": 1,
        "sun_exposure": 0,
        "hygiene_issue": 0,
        "wbc": 4000,
        "platelets": 100000,
        "haemoglobin": 13.5
    }

    print("Waiting for API to be ready...")
    for _ in range(30):
        try:
            requests.get("http://localhost:8000/")
            break
        except requests.RequestException:
            time.sleep(1)
    else:
        print("API failed to start.")
        sys.exit(1)

    print("Sending prediction request...")
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        print("Prediction Result:", result)

        if all(key in result for key in ["prediction", "probabilities", "severity", "advice"]):
            print("Test PASSED")
        else:
            print("Test FAILED: Invalid response format")
            sys.exit(1)

    except Exception as e:
        print(f"Test FAILED: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_api()
