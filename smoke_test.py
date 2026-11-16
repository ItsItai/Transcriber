# Run after upgrading torch/transformers: python smoke_test.py  (needs the GPU and ffmpeg, ~1 min)
# Guards the transformers internals app.py relies on: one generate() call per batch, ignore_warning, dtype.
import os
import subprocess
import tempfile
import threading

import app

calls = []
app.window = type("W", (), {"evaluate_js": lambda self, js: calls.append(js)})()
app.loader = threading.Thread(target=app.load_model)
app.loader.start()

with tempfile.TemporaryDirectory() as d:
    clip = os.path.join(d, "clip.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=300:duration=75",
            clip,
        ],
        check=True,
    )
    app.transcribe(clip)

assert any(c.startswith("done(") for c in calls), calls[-1]
assert any('"Transcribing\\u2026", 0.99' in c for c in calls), (
    "progress never advanced: generate() hook not called"
)
assert os.path.exists(os.path.join(app.OUT_DIR, "clip.txt"))
app.clear_transcripts()
print("ok")
