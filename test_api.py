import urllib.request
import json

# Replace with your actual live Render web service URL (e.g., https://luna-xyz.onrender.com/chat)
url = "https://YOUR-RENDER-URL-HERE.onrender.com/chat"

payload = {"message": "Hello Luna, how are you?"}
data = json.dumps(payload).encode("utf-8")

req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")

try:
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode("utf-8"))
        print("Luna Response:", result)
except Exception as e:
    print("Error connecting to live API:", e)