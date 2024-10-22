import requests
import os
import mimetypes
from datetime import datetime

url = "https://api.sarvam.ai/speech-to-text-translate" 
file_path = r"C:\Users\dsupr\OneDrive\Documents\Audacity\constitution_audio_noise_cropped_final_2min.wav"
headers = {
    "api-subscription-key": "aa2dcba2-0073-49d4-bce8-4489d118f7b4",
}

try:
    with open(file_path, "rb") as audio_file:
        file_name = os.path.basename(file_path)
        files = {
            'file': (file_name, audio_file,'audio/wav') 
        }
        tstart=datetime.now()
        # Send the POST request
        response = requests.post(url, files=files, headers=headers)
        tend=datetime.now()
    print("Response Text:", response.text)
    print(tend-tstart)
except Exception as e:
    print("An error occurred:", e)
