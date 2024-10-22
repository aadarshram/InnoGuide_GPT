import os
from dotenv import load_dotenv
import requests
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydub import AudioSegment
import io

# Load environment variables from .env file
load_dotenv()

# Function to make API request for text-to-speech
def generate_audio(text_chunk):
    url = "https://api.sarvam.ai/text-to-speech"
    
    payload = {
        "inputs": [text_chunk],
        "target_language_code": "hi-IN",
        "enable_preprocessing": True
    }
    
    headers = {
        "api-subscription-key": os.getenv("SARVAM_API_KEY"),
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        response_data = response.json()
        
        if "audios" in response_data and len(response_data["audios"]) > 0:
            return base64.b64decode(response_data["audios"][0])
        else:
            print("Error: No audio generated for the chunk.")
            return None
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None

# Function to split text into chunks
def split_text(text, max_chunk_size=200):
    words = text.split()
    chunk_size = len(words) / 3
    chunks = []
    current_chunk = []

    for word in words:
        # Calculate the current chunk size including the new word
        chunk_length = sum(len(w) for w in current_chunk) + len(word) + len(current_chunk)  # Adding spaces

        if chunk_length > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]  # Start a new chunk with the current word
        else:
            current_chunk.append(word)

    if current_chunk:
        chunks.append(" ".join(current_chunk))  # Add any remaining words as a chunk

    return chunks

# Main function to orchestrate the audio generation
def main():
    text = "This is a sample text to be converted to speech. The text will be split into multiple chunks and each chunk will be converted to audio separately."
    
    text_chunks = split_text(text)  # Split the text into chunks

    audio_segments = [None] * len(text_chunks)

    with ThreadPoolExecutor(max_workers = min(10, len(text_chunks))) as executor:
        futures = {executor.submit(generate_audio, chunk): chunk for chunk in text_chunks}
        
        for future in as_completed(futures):
            audio_data = future.result()
            chunk = futures[future]
            index = text_chunks.index(chunk)  # Find the index of the chunk

            if audio_data:
                audio_segments[index] = AudioSegment.from_wav(io.BytesIO(audio_data))  # Convert bytes to AudioSegment

    # Combine all audio segments
    if audio_segments:
        final_audio = AudioSegment.silent(duration=0)  # Start with silent audio
        for segment in audio_segments:
            if segment is not None:  # Check if segment is not None
                final_audio += segment  # Concatenate audio segments

        output_filename = "output_audio.wav"
        final_audio.export(output_filename, format="wav")
        print(f"Audio saved as {output_filename}")
    else:
        print("No audio segments generated.")

if __name__ == "__main__":
    main()
