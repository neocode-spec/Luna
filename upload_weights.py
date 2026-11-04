import base64
import os
import urllib.request
import json

# Replace with your actual GitHub repo details
owner = "neocode-spec"  # Your GitHub username/org from your repository URL
repo = "Luna"           # Your repository name
path = "luna_1.4_weights.pt"
print("Reading compressed weights file...")
with open(path, "rb") as f:
    file_content = f.read()

encoded_content = base64.b64encode(file_content).decode("utf-8")
url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"

# Check if file already exists to get its SHA (required for updates)
req = urllib.request.Request(url, headers={"Authorization": f"token {token}", "User-Agent": "Python"})
sha = None
try:
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
        sha = data.get("sha")
except Exception:
    pass

# Prepare payload
payload = {
    "message": "Add compressed Luna 1.4 weights via Python API",
    "content": encoded_content
}
if sha:
    payload["sha"] = sha

data_json = json.dumps(payload).encode("utf-8")
put_req = urllib.request.Request(url, data=data_json, headers={
    "Authorization": f"token {token}",
    "User-Agent": "Python",
    "Content-Type": "application/json"
}, method="PUT")

print("Uploading weights to GitHub...")
with urllib.request.urlopen(put_req) as response:
    print("Success! Weights uploaded to GitHub repository.")