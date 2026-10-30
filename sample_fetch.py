#!/usr/bin/env python3
"""
Download and prepare exactly 10 evaluation samples for MTE252 Phase 1 Task 2.
Categories: 2 female, 2 male, 2 child, 2 multi-speaker, 2 noisy.
Target: 30-60 s, >=16 kHz, 16-bit, mono WAV.

Genders come from LibriSpeech SPEAKERS.TXT, not file order.
Child clips are ENNI story retells (child turns only).
Multi-speaker clips alternate a male reader and a female reader.
If a sample wav is already in audio_sampling/tmp_audio/, it is left as-is and not fetched again.
"""

import io
import os
import sys
import tarfile
import zipfile
import urllib.request
from pathlib import Path

try:
    import soundfile as sf
    import numpy as np
    from scipy import signal
    from tqdm import tqdm
except ImportError:
    print("Installing required packages: soundfile scipy numpy tqdm")
    os.system("pip install soundfile scipy numpy tqdm -q")
    import soundfile as sf
    import numpy as np
    from scipy import signal
    from tqdm import tqdm

TMP = Path("tmp_audio")
TMP.mkdir(exist_ok=True)
OUT_DIR = Path("audio_sampling") / "tmp_audio"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SR = 16000
UA = {"User-Agent": "Mozilla/5.0 (compatible; MTE252-sample-fetch/1.0)"}

# LibriSpeech dev-clean reader ids. Sex is from SPEAKERS.TXT.
# 1462 was confirmed by ear as the female voice previously saved as 05/06.
# 1272 was confirmed by ear as the male voice in the clean male samples.
FEMALE_ID = "1462"
MALE_ID = "1272"
# Distinct from the single-speaker samples, so each mix is a new pair.
MULTI_PAIRS = (("174", "84"), ("251", "3536"))  # (male, female)
# ENNI session ids. CHI turns only; examiner turns are dropped.
CHILD_SESSIONS = ("801", "902")

SAMPLE_NAMES = [
    "01_female_clean_1.wav",
    "02_female_clean_2.wav",
    "03_male_clean_1.wav",
    "04_male_clean_2.wav",
    "05_child_approx_1.wav",
    "06_child_approx_2.wav",
    "07_multi_speaker_1.wav",
    "08_multi_speaker_2.wav",
    "09_noisy_1.wav",
    "10_noisy_2.wav",
]


class _OneLine:
    """Keep tqdm on a single terminal line (IDE terminals often wrap or ignore \\r)."""

    def __init__(self, stream):
        self._s = stream

    def write(self, text):
        if not text:
            return 0
        if text.startswith("\r"):
            text = "\r\033[2K" + text[1:]
        self._s.write(text)
        return len(text)

    def flush(self):
        self._s.flush()

    def isatty(self):
        return True

    def fileno(self):
        return self._s.fileno()


def pbar(*args, **kwargs):
    kwargs.setdefault("file", _OneLine(sys.stderr))
    kwargs.setdefault("ncols", 72)
    kwargs.setdefault("mininterval", 0.5)
    kwargs.setdefault("leave", True)
    kwargs.setdefault("ascii", True)
    kwargs.setdefault("dynamic_ncols", False)
    return tqdm(*args, **kwargs)


def have_sample(name):
    path = OUT_DIR / name
    return path.exists() and path.stat().st_size > 0


def download(url, dest):
    """Download url to dest. Skip when dest is already a complete file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = dest.stat().st_size if dest.exists() else 0
    expected = 0
    if size > 0:
        try:
            req = urllib.request.Request(url, method="HEAD", headers=UA)
            with urllib.request.urlopen(req) as r:
                expected = int(r.headers.get("Content-Length") or 0)
        except Exception:
            expected = 0
        if not expected or size == expected:
            tqdm.write(f"Already have {dest}")
            return
        tqdm.write(f"Truncated {dest} ({size}/{expected}), re-downloading")
    if dest.exists():
        dest.unlink()
    req = urllib.request.Request(url, headers=UA)
    resp = urllib.request.urlopen(req)
    total = int(resp.headers.get("Content-Length") or 0)
    with open(dest, "wb") as out, pbar(
        total=total or None,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=f"Download {dest.name}",
    ) as bar:
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            out.write(chunk)
            bar.update(len(chunk))
    got = dest.stat().st_size
    if total and got != total:
        dest.unlink()
        raise OSError(f"Incomplete download {dest} ({got}/{total})")
    tqdm.write(f"Saved {dest}")


def resample_to_16k(data, sr):
    if sr == SR:
        return np.asarray(data, dtype=np.float32)
    num = int(len(data) * SR / sr)
    return signal.resample(data, num).astype(np.float32)


def read_mono(path):
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return resample_to_16k(data, sr)


def take_audio(files, seconds):
    pieces = []
    total = 0.0
    for f in files:
        data = read_mono(f)
        pieces.append(data)
        total += len(data) / SR
        if total >= seconds:
            break
    if not pieces:
        raise RuntimeError("no audio in file list")
    audio = np.concatenate(pieces)
    return audio[: int(seconds * SR)]


def save_wav(path, audio):
    audio = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    # Trim a trailing all-zero pad so the file is speech, not silence.
    nonzero = np.flatnonzero(np.abs(audio) > 1e-4)
    if len(nonzero):
        audio = audio[: nonzero[-1] + 1]
    sf.write(path, audio, SR, subtype="PCM_16")
    tqdm.write(f"Created {path} ({len(audio) / SR:.1f}s)")


def speaker_files(libri_root, speaker_id):
    files = sorted(
        p for p in libri_root.rglob("*.flac") if p.parent.parent.name == speaker_id
    )
    if not files:
        raise RuntimeError(f"No LibriSpeech audio for speaker {speaker_id}")
    return files


def fade_edges(chunk, fade=128):
    if len(chunk) <= 2 * fade:
        return chunk
    out = chunk.copy()
    ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
    out[:fade] *= ramp
    out[-fade:] *= ramp[::-1]
    return out


def alternate_speakers(male_files, female_files, target_sec, turn_sec=6.0):
    """Take turns so both speakers are audible from the start of the file."""
    male = take_audio(male_files, target_sec)
    female = take_audio(female_files, target_sec)
    turn = int(turn_sec * SR)
    gap = np.zeros(int(0.25 * SR), dtype=np.float32)
    pieces = []
    cursors = [0, 0]
    sources = (male, female)
    which = 0
    total = 0.0
    while total < target_sec:
        src = sources[which]
        start = cursors[which]
        if start >= len(src) - SR:
            break
        chunk = fade_edges(src[start : start + turn])
        cursors[which] = start + turn
        pieces.append(chunk)
        total += len(chunk) / SR
        if total >= target_sec:
            break
        pieces.append(gap)
        total += len(gap) / SR
        which = 1 - which
    if cursors[0] < SR or cursors[1] < SR:
        raise RuntimeError("multi-speaker clip is missing one of the speakers")
    audio = np.concatenate(pieces)
    return audio[: int(target_sec * SR)]


def read_parquet(path):
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("Installing required package: pyarrow")
        os.system("pip install pyarrow -q")
        import pyarrow.parquet as pq
    return pq.read_table(path)


def enni_child(session_id, target_sec):
    """Child-only turns from one ENNI session, in story order."""
    dest = TMP / "enni" / f"{session_id}.parquet"
    url = (
        "https://huggingface.co/datasets/TomRoma/ENNI_dataset/resolve/main/data/"
        f"{session_id}-00000-of-00001.parquet"
    )
    download(url, dest)
    table = read_parquet(dest)
    rows = []
    for label, start, audio in zip(
        table.column("speaker_label").to_pylist(),
        table.column("start_ms").to_pylist(),
        table.column("audio").to_pylist(),
    ):
        if label != "CHI":
            continue
        rows.append((start, audio["bytes"]))
    rows.sort()
    pieces = []
    total = 0.0
    gap = np.zeros(int(0.18 * SR), dtype=np.float32)
    for _, raw in rows:
        data, sr = sf.read(io.BytesIO(raw))
        if getattr(data, "ndim", 1) > 1:
            data = data.mean(axis=1)
        data = resample_to_16k(data, sr)
        if len(data) < int(0.25 * SR):
            continue
        if pieces:
            pieces.append(gap)
            total += len(gap) / SR
        pieces.append(fade_edges(data))
        total += len(data) / SR
        if total >= target_sec:
            break
    if total < 30:
        raise RuntimeError(f"ENNI session {session_id} has only {total:.1f}s of child speech")
    audio = np.concatenate(pieces)
    return audio[: int(target_sec * SR)]


def ensure_librispeech():
    found = list(TMP.rglob("dev-clean-2"))
    libri_root = found[0] if found else TMP / "LibriSpeech" / "dev-clean-2"
    if libri_root.exists() and any(libri_root.rglob("*.flac")):
        tqdm.write("Already extracted Mini LibriSpeech")
        return libri_root
    mini_url = "https://www.openslr.org/resources/31/dev-clean-2.tar.gz"
    mini_tar = TMP / "dev-clean-2.tar.gz"
    download(mini_url, mini_tar)
    with tarfile.open(mini_tar, "r:gz") as tar:
        members = tar.getmembers()
        for member in pbar(members, desc="Extract LibriSpeech", unit="file"):
            tar.extract(member, TMP)
    found = list(TMP.rglob("dev-clean-2"))
    libri_root = found[0] if found else TMP / "LibriSpeech" / "dev-clean-2"
    return libri_root


def ensure_noisy():
    dest_root = TMP / "valentini"
    noisy = sorted(dest_root.rglob("*.wav"))
    if noisy:
        tqdm.write(f"Already extracted Valentini ({len(noisy)} WAV)")
        return noisy
    val_url = "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_testset_wav.zip?sequence=4&isAllowed=y"
    val_zip = TMP / "noisy_testset_wav.zip"
    download(val_url, val_zip)
    with zipfile.ZipFile(val_zip, "r") as z:
        names = z.namelist()
        for name in pbar(names, desc="Extract Valentini", unit="file"):
            z.extract(name, dest_root)
    return sorted(dest_root.rglob("*.wav"))


def build_clean(libri_root):
    female = speaker_files(libri_root, FEMALE_ID)
    male = speaker_files(libri_root, MALE_ID)
    mid_f = max(1, len(female) // 2)
    mid_m = max(1, len(male) // 2)
    jobs = (
        ("01_female_clean_1.wav", female[:mid_f], 40),
        ("02_female_clean_2.wav", female[mid_f:], 42),
        ("03_male_clean_1.wav", male[:mid_m], 38),
        ("04_male_clean_2.wav", male[mid_m:], 45),
    )
    for name, files, seconds in jobs:
        path = OUT_DIR / name
        if have_sample(name):
            tqdm.write(f"Already have {path}")
            continue
        save_wav(path, take_audio(files, seconds))


def build_multi(libri_root):
    jobs = (
        ("07_multi_speaker_1.wav", MULTI_PAIRS[0], 42),
        ("08_multi_speaker_2.wav", MULTI_PAIRS[1], 45),
    )
    for name, (male_id, female_id), seconds in jobs:
        path = OUT_DIR / name
        if have_sample(name):
            tqdm.write(f"Already have {path}")
            continue
        audio = alternate_speakers(
            speaker_files(libri_root, male_id),
            speaker_files(libri_root, female_id),
            seconds,
        )
        save_wav(path, audio)


def build_child():
    jobs = (
        ("05_child_approx_1.wav", CHILD_SESSIONS[0], 40),
        ("06_child_approx_2.wav", CHILD_SESSIONS[1], 42),
    )
    for name, session_id, seconds in jobs:
        path = OUT_DIR / name
        if have_sample(name):
            tqdm.write(f"Already have {path}")
            continue
        save_wav(path, enni_child(session_id, seconds))


def build_noisy():
    names = ("09_noisy_1.wav", "10_noisy_2.wav")
    if all(have_sample(n) for n in names):
        for name in names:
            tqdm.write(f"Already have {OUT_DIR / name}")
        return
    noisy = ensure_noisy()
    if len(noisy) < 2:
        raise RuntimeError("Valentini noisy set is missing")
    jobs = (
        ("09_noisy_1.wav", noisy[0:8], 40),
        ("10_noisy_2.wav", noisy[8:16], 42),
    )
    for name, files, seconds in jobs:
        path = OUT_DIR / name
        if have_sample(name):
            tqdm.write(f"Already have {path}")
            continue
        save_wav(path, take_audio(files, seconds))


def check_samples():
    """Fail if any sample is missing or not playable 16-bit mono >=16 kHz, 30-60 s."""
    for name in SAMPLE_NAMES:
        path = OUT_DIR / name
        info = sf.info(path)
        if info.samplerate < SR or info.channels != 1 or info.subtype != "PCM_16":
            raise SystemExit(
                f"{name} is {info.samplerate} Hz, {info.channels} ch, {info.subtype}"
            )
        dur = info.duration
        if not (30 <= dur <= 60):
            raise SystemExit(f"{name} is {dur:.1f}s, expected 30-60s")
        print(f"OK {name}: {info.samplerate} Hz mono {info.subtype} {dur:.1f}s")


def main():
    clean_or_multi = [
        "01_female_clean_1.wav",
        "02_female_clean_2.wav",
        "03_male_clean_1.wav",
        "04_male_clean_2.wav",
        "07_multi_speaker_1.wav",
        "08_multi_speaker_2.wav",
    ]
    if any(not have_sample(n) for n in clean_or_multi):
        libri_root = ensure_librispeech()
        build_clean(libri_root)
        build_multi(libri_root)
    else:
        for name in clean_or_multi:
            tqdm.write(f"Already have {OUT_DIR / name}")

    build_child()
    build_noisy()
    check_samples()
    print("\nDone. 10 samples are in the 'audio_sampling/tmp_audio/' folder.")


if __name__ == "__main__":
    main()
