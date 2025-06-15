import os
import sounddevice as sd
from scipy.io.wavfile import write
from faster_whisper import WhisperModel

# Initialize the Whisper model once (it’s heavy to reload each time)
# Use "base.en" or another local model you’ve downloaded
_model = WhisperModel("base.en", device="auto")

BASE_AUDIO_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data", "audio")

def record_audio(filename: str = "input.wav", duration: int = 5, fs: int = 16000):
    """
    Records `duration` seconds and saves to data/audio/filename.
    """
    # Make sure the folder exists
    os.makedirs(BASE_AUDIO_DIR, exist_ok=True)

    filepath = os.path.join(BASE_AUDIO_DIR, filename)
    print(f"Recording {duration}s audio to '{filepath}' …")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filepath, fs, recording)
    print("Recording complete.")
    return filepath  # return the path so caller can transcribe it

def transcribe_audio(filename: str):
    """
    Runs Whisper on the given file and returns the transcript.
    """
    print(f"Transcribing '{filename}' …")
    segments, _ = _model.transcribe(filename, beam_size=5)
    text = "".join(seg.text for seg in segments)
    print("Transcription complete:", text)
    return text

if __name__ == "__main__":
    wav = record_audio(duration=3)
    print(transcribe_audio(wav))
