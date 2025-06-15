# src/run_robot_brain.py

import os
import threading
import pyttsx3
import openai

from .audio_io import record_audio, transcribe_audio
from .memory import (
    initialize_memory_store,
    query_memories,
    add_memory,
    decay_memories,
    flag_for_summarization
)

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

# ─── Memory Database Initialization & Background Tasks ────────────────────────────
# Path for REMI’s memory store
DB_PATH = os.path.join(project_root, "data", "memory", "remi_memory.db")
initialize_memory_store(DB_PATH)

def schedule_background_tasks(interval_hours: float = 1.0):
    """
    Periodically decay and flag old memories for summarization.
    Runs every `interval_hours` hours in a background thread.
    """
    # 1) Decay memory strengths
    decay_memories(DB_PATH)
    # 2) Flag weak/unemotional memories for summarization
    flag_for_summarization(DB_PATH)
    # 3) Reschedule
    threading.Timer(interval_hours * 3600, schedule_background_tasks, args=[interval_hours]).start()

# Kick off the hourly background job
schedule_background_tasks()

# ─── Model Query ────────────────────────────────────────────────────────────────────
def ask_model(prompt: str) -> str:
    """
    Send the user prompt to the OpenAI Chat API and return the assistant's reply.
    """
    completion = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": prompt}],
        temperature=0.7,
        max_tokens=256
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
    print("REMI brain starting. Press Ctrl+C to exit.")
    while True:
        # 1) Record and save audio (3s by default)
        wav_path = record_audio(filename="input.wav", duration=3)

        # 2) Transcribe the recorded file
        user_input = transcribe_audio(wav_path)
        if not user_input.strip():
            print("No speech detected, listening again…")
            continue

        # 3) Retrieve relevant memories
        memories = query_memories(DB_PATH, user_input, top_k=5)
        # Build a context block from those memories
        mem_block = ""
        for item in memories:
            entry = item["entry"]
            mem_block += f"[On {entry.timestamp.date()}] You said: \"{entry.content}\".\n"
        
        # 4) Ask the model, injecting memory context
        prompt = mem_block + "\nUser: " + user_input
        response = ask_model(prompt)
        print("REMI:", response)

        # 5) Speak the response aloud
        speak(response)

        # 6) Store this turn as new episodic memories
        add_memory(DB_PATH, f"User: {user_input}",   category="episodic", emotions={})
        add_memory(DB_PATH, f"REMI: {response}",     category="episodic", emotions={})

if __name__ == "__main__":
    main()
