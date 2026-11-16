# Usage: python app.py   -> desktop window; writes transcripts/<video>.txt, cleared on next run and on exit
import glob
import json
import logging
import math
import os
import subprocess
import threading
import winreg
from contextlib import suppress

import numpy as np
import webview

# A terminal opened before ffmpeg was installed has a stale PATH; append the current one from the registry.
for root, key in (
    (winreg.HKEY_CURRENT_USER, "Environment"),
    (
        winreg.HKEY_LOCAL_MACHINE,
        r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
    ),
):
    with suppress(OSError), winreg.OpenKey(root, key) as k:
        os.environ["PATH"] += os.pathsep + os.path.expandvars(
            winreg.QueryValueEx(k, "Path")[0]
        )

SR = 16000
CHUNK_S, STEP_S, BATCH = (
    30,
    20,
    8,
)  # pipeline strides 5 s each side of a 30 s chunk -> advances 20 s
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "transcripts")
pipe = None
load_error = None

# Noise we can't fix at the source; each filter drops only that one message.
for name, text in (
    (
        "torch.distributed.elastic.multiprocessing.redirects",
        "Redirects are currently not supported",
    ),  # irrelevant to us
    (
        "transformers.generation.utils",
        "together with generation-related",
    ),  # raised by Whisper's own generate()
):
    logging.getLogger(name).addFilter(lambda r, t=text: t not in r.getMessage())


def load_model():
    global pipe, load_error
    try:
        import torch
        from transformers import pipeline

        pipe = pipeline(
            "automatic-speech-recognition",
            model="ivrit-ai/whisper-large-v3-turbo",
            dtype=torch.float16,
            device="cuda:0",
        )
        pipe.tokenizer.clean_up_tokenization_spaces = (
            False  # transformers skips it for Whisper anyway, and warns
        )
    except Exception as e:  # noqa: BLE001 - thread top level: any load error must reach the UI
        load_error = e


def clear_transcripts():
    # The folder belongs to the app, so every .txt in it is a previous result.
    for f in glob.glob(os.path.join(OUT_DIR, "*.txt")):
        with suppress(OSError):
            os.remove(f)


def ui(fn, *args):
    window.evaluate_js(f"{fn}(...{json.dumps(args)})")


def transcribe(path):
    try:
        clear_transcripts()
        if loader.is_alive():
            ui("status", "Loading model… (first run downloads ~1.6 GB)", None)
            loader.join()
        if pipe is None:
            load_model()  # retry: the first load may have hit a passing problem (offline, GPU busy)
        if pipe is None:
            raise RuntimeError(f"Model failed to load: {load_error}")

        ui("status", "Extracting audio…", None)
        proc = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-i",
                path,
                "-vn",
                "-f",
                "f32le",
                "-ac",
                "1",
                "-ar",
                str(SR),
                "-",
            ],
            capture_output=True,
            check=False,  # returncode handled below to show ffmpeg's own error
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        if proc.returncode:
            raise RuntimeError(
                (
                    proc.stderr.decode(errors="replace").strip().splitlines()
                    or ["ffmpeg failed"]
                )[-1]
            )
        audio = np.frombuffer(proc.stdout, np.float32)
        duration = len(audio) / SR

        # The pipeline calls model.generate once per batch of chunks; count calls to drive the bar.
        n_chunks = max(1, math.ceil((duration - CHUNK_S) / STEP_S) + 1)
        total, done = math.ceil(n_chunks / BATCH), 0
        generate = pipe.model.generate

        def counted(*a, **k):
            nonlocal done
            out = generate(*a, **k)
            done += 1
            ui("status", "Transcribing…", min(done / total, 0.99))
            return out

        ui("status", "Transcribing…", 0)
        pipe.model.generate = counted
        try:
            out = pipe(
                {"raw": audio, "sampling_rate": SR},
                chunk_length_s=CHUNK_S,
                ignore_warning=True,  # chunked mode is a deliberate choice: batched chunks are what make it fast
                batch_size=BATCH,
                generate_kwargs={"language": "he", "task": "transcribe"},
            )
        finally:
            del pipe.model.generate  # back to the class method

        text = out["text"].strip()
        os.makedirs(OUT_DIR, exist_ok=True)
        txt = os.path.join(
            OUT_DIR, os.path.splitext(os.path.basename(path))[0] + ".txt"
        )
        with open(txt, "w", encoding="utf-8") as f:
            f.write(text)
        ui("done", text, txt)
    except FileNotFoundError:
        ui("fail", "ffmpeg was not found. Install it and make sure it is on PATH.")
    except Exception as e:  # noqa: BLE001 - thread top level: otherwise the UI hangs on the progress screen
        ui("fail", str(e) or type(e).__name__)


class Api:
    def pick(self):
        r = window.create_file_dialog(
            webview.FileDialog.OPEN,
            file_types=(
                "Media files (*.mp4;*.mkv;*.mov;*.avi;*.webm;*.mp3;*.wav;*.m4a)",
                "All files (*.*)",
            ),
        )
        return r[0] if r else None

    def start(self, path):
        threading.Thread(target=transcribe, args=(path,), daemon=True).start()


if __name__ == "__main__":
    loader = threading.Thread(target=load_model, daemon=True)
    loader.start()  # warm the model while the user picks a file
    window = webview.create_window(
        "Transcriber",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui.html"),
        js_api=Api(),
        width=1000,
        height=1000,
        resizable=False,  # also removes the maximize button
        # pywebview's default centering is ignored on Windows; naming the screen makes it compute the center itself.
        screen=next(
            s for s in webview.screens if s.x == s.y == 0
        ),  # primary monitor sits at 0,0
    )
    webview.start()
    clear_transcripts()  # window closed
