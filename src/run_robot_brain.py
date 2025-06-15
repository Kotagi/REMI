# src/run_robot_brain.py

import os
import pyttsx3
import openai

from .audio_io import record_audio, transcribe_audio

# ─── Locate & Load OpenAI API Key ─────────────────────────────────────────────────
# Project root is one level above src/
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Primary expected location now under "configs"
key_file = os.path.join(project_root, "configs", "api_key", "openai_key.txt")

api_key = None

# Try loading from file
if os.path.exists(key_file):
    with open(key_file, "r") as f:
        api_key = f.read().strip()
elif os.getenv("OPENAI_API_KEY"):
    # Fallback to environment variable
    api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise FileNotFoundError(
        f"OpenAI API key not found!\n\n"
        f"Tried file: {key_file}\n"
        f"And environment variable: OPENAI_API_KEY\n"
        f"Please place your key in that file or set the env var."
    )

openai.api_key = api_key

# ─── Model Query ────────────────────────────────────────────────────────────────────
def ask_model(prompt: str) -> str:
    """
    Send the user prompt to the OpenAI Chat API and return the assistant's reply.
    """
    completion = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=128
    )
    return completion.choices[0].message.content.strip()

# ─── Text-to-Speech ─────────────────────────────────────────────────────────────────
def speak(text: str):
    """
    Convert text to speech and play it.
    """
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)
    engine.say(text)
    engine.runAndWait()

# ─── Main Loop ─────────────────────────────────────────────────────────────────────
def main():
    print("Robot brain starting. Press Ctrl+C to exit.")
    while True:
        # 1) Record and save audio
        wav_path = record_audio(filename="input.wav", duration=3)

        # 2) Transcribe the recorded file
        user_input = transcribe_audio(wav_path)
        if not user_input.strip():
            print("No speech detected.")
            continue

        # 3) Get the model’s response
        response = ask_model(user_input)
        print("Bot:", response)

        # 4) Speak the response aloud
        speak(response)

if __name__ == "__main__":
    main()
