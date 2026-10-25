
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

SR = 22050
RAW, CLEAN = Path("recordings_raw"), Path("recordings_clean")
CLEAN.mkdir(exist_ok=True)

total = 0.0
good = []
problems = []

print(f"{'file':10s} {'sec':>5s} {'peak':>5s} {'level dB':>9s}  flags")
for f in sorted(RAW.glob("*.wav")):
    y, _ = librosa.load(f, sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=30)
    if len(y) == 0:
        print(f"{f.name:10s} EMPTY")
        problems.append(f.name)
        continue

    peak = float(np.abs(y).max())
    dur = len(y) / SR
    level = 20 * np.log10(np.sqrt((y ** 2).mean()) + 1e-9)

    flags = []
    if peak >= 0.99:
        flags.append("clipping")
    if peak < 0.1:
        flags.append("too quiet")
    if dur < 1.5:
        flags.append("very short")
    if dur > 15:
        flags.append("very long")

    y = y / (peak + 1e-9) * 0.9
    sf.write(CLEAN / f.name, y, SR)
    total += dur
    print(f"{f.name:10s} {dur:5.1f} {peak:5.2f} {level:9.1f}  {', '.join(flags)}")

    if flags:
        problems.append(f.name)
    elif 3 <= dur <= 10:
        good.append(y)

print(f"\nTotal speech after trimming: {total / 60:.1f} minutes ({total:.0f} s)")
print(f"Clips with flags (listen to these, redo if bad): {problems or 'none'}")

# Reference clip for zero-shot cloning: about 25 s of clean clips with short gaps
gap = np.zeros(int(0.3 * SR), dtype=np.float32)
ref, length = [], 0.0
for y in good:
    ref.extend([y.astype(np.float32), gap])
    length += len(y) / SR + 0.3
    if length >= 25:
        break

if ref:
    sf.write("reference.wav", np.concatenate(ref), SR)
    print(f"reference.wav written ({length:.0f} s)")
else:
    print("No clean clips of 3-10 s found for reference.wav. Check the flags above.")