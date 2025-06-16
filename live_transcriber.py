import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
import queue

# Setup audio recording
samplerate = 16000
blocksize = 4000
q = queue.Queue()

def callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(indata.copy())

# Load Whisper model
model = WhisperModel("tiny", device="cpu", compute_type="int8")

print("🎤 Listening... Press Ctrl+C to stop.\n")

# Start streaming
with sd.InputStream(samplerate=samplerate, channels=1, blocksize=blocksize, callback=callback):
    audio_buffer = np.empty((0, 1), dtype=np.float32)

    while True:
        try:
            data = q.get()
            audio_buffer = np.concatenate((audio_buffer, data))

            # Process every ~1.5 seconds of audio
            if len(audio_buffer) >= samplerate * 1.5:
                slice_len = int(samplerate * 1.5)
                segment = audio_buffer[:slice_len]
                audio_buffer = audio_buffer[slice_len:]

                segment = segment.flatten()
                segments, _ = model.transcribe(segment, language="en")
                
                for seg in segments:
                    print(f">> {seg.text}")

        except KeyboardInterrupt:
            print("\n🛑 Stopped.")
            break
