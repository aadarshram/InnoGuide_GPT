import requests
import base64
url = "https://api.sarvam.ai/text-to-speech"

payload = {
    "inputs": ["So, schedules are basically tables that contain some additional details that are not present in articles. For example, Article 346 and 347 of the Indian Constitution talk about official languages but do not specify which languages will be official and which will not. This tells about the poor age schedule, so additional details of the schedule are given. Now, there is a very simple trick to remember these twelve schedules, if you know, you will never forget. This is the Tears of Old Sea, it has twelve letters and twelve schedules, meaning each letter is for a schedule. See how. Territories, involving affirmations, Raj Bhavan, schedule letters, so each letter is a schedule letter, simple, easy to remember and share"],
    "target_language_code": "hi-IN",
    "enable_preprocessing":True
}
headers = {
    "api-subscription-key": "aa2dcba2-0073-49d4-bce8-4489d118f7b4",
    "Content-Type": "application/json"
}

response = requests.request("POST", url, json=payload, headers=headers)

if response.status_code == 200:
    response_data = response.json()
    
    if "audios" in response_data and len(response_data["audios"]) > 0:
        base64_string = response_data["audios"][0] 
        output_filename="output_audio.wav"
        wav_bytes = base64.b64decode(base64_string)
        with open(output_filename, "wb") as wav_file:
            wav_file.write(wav_bytes)
        print(f"Audio saved as {output_filename}")
    else:
        print("Error:No audio generated in the response.")
else:
    print(f"Error: {response.status_code}, {response.text}")
