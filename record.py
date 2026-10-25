"""Record one WAV clip per sentence in script.txt.

Usage (inside the voiceenv environment):
    python record.py

For each sentence: press Enter, speak, press Enter again to stop.
Choose 'r' to redo a take. Existing clips are skipped, so you can stop
and continue later. A metadata.csv (filename|text) is written at the end.
"""
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf

SR = 22050
OUT = Path("recordings_raw")
OUT.mkdir(exist_ok=True)

lines = [l.strip() for l in open("script.txt", encoding="utf-8") if l.strip()]
print(f"{len(lines)} sentences. Default microphone: {sd.query_devices(kind='input')['name']}")

for i, text in enumerate(lines, 1):
    path = OUT / f"s{i:03d}.wav"
    if path.exists():
        continue
    while True:
        print(f"\n[{i}/{len(lines)}] {text}")
        input("Press Enter to start, speak, then press Enter to stop... ")
        chunks = []

        def callback(indata, frames, time, status):
            chunks.append(indata.copy())

        with sd.InputStream(samplerate=SR, channels=1, dtype="float32", callback=callback):
            input()
        audio = np.concatenate(chunks)
        sf.write(path, audio, SR)
        print(f"Saved {path.name} ({len(audio) / SR:.1f} s)")
        if input("Press Enter to keep it, or type r to redo: ").strip().lower() != "r":
            break

with open("metadata.csv", "w", encoding="utf-8") as f:
    for i, text in enumerate(lines, 1):
        if (OUT / f"s{i:03d}.wav").exists():
            f.write(f"s{i:03d}.wav|{text}\n")
print("\nDone. metadata.csv written.")