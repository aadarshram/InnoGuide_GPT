import requests
import os
# Replace with your actual OpenAI API key and organization ID
api_key = os.getenv("OPENAI_API_KEY")
organization_id = os.getenv("ORG_KEY")

# API endpoint
url = 'https://api.openai.com/v1/chat/completions'

# Sample data to send
data = {
    "model": "gpt-3.5-turbo",  # or another model you're authorized to use
    "messages": [{"role": "user", "content": "Hello, OpenAI!"}],
}

# Set up the headers
headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json',
    'OpenAI-Organization': organization_id
}

# Make the request
response = requests.post(url, headers=headers, json=data)

# Check the response
if response.status_code == 200:
    print("API key is working!")
    print("Response:", response.json())
else:
    print("Failed to connect.")
    print("Status code:", response.status_code)
    print("Response:", response.text)
