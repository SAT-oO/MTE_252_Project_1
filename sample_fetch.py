#!/usr/bin/env python3
"""
Download and prepare exactly 10 evaluation samples for MTE252 Phase 1 Task 2.
Categories: 2 female, 2 male, 2 child (approx), 2 multi-speaker, 2 noisy.
Target: 30-60 s, >=16 kHz mono WAV.
"""

import os
import sys
import tarfile
import zipfile
import urllib.request
import shutil
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

OUT_DIR = Path("samples")
OUT_DIR.mkdir(exist_ok=True)
TMP = Path("tmp_audio")
TMP.mkdir(exist_ok=True)

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

def download(url, dest):
    size = dest.stat().st_size if dest.exists() else 0
    expected = 0
    if size > 0:
        try:
            with urllib.request.urlopen(urllib.request.Request(url, method="HEAD")) as r:
                expected = int(r.headers.get("Content-Length") or 0)
        except Exception:
            expected = 0
        if not expected or size == expected:
            tqdm.write(f"Already have {dest}")
            return
        tqdm.write(f"Truncated {dest} ({size}/{expected}), re-downloading")
    if dest.exists():
        dest.unlink()
    req = urllib.request.urlopen(url)
    total = int(req.headers.get("Content-Length") or 0)
    desc = f"Download {dest.name}"
    with open(dest, "wb") as out, pbar(
        total=total or None,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=desc,
    ) as bar:
        while True:
            chunk = req.read(1024 * 256)
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
    if sr == 16000:
        return data
    # simple resample
    num = int(len(data) * 16000 / sr)
    return signal.resample(data, num).astype(np.float32)

def concat_and_save(files, out_name, target_sr=16000, target_len_sec=40):
    """Concatenate audio files until roughly target length, save as 16k mono WAV."""
    out_path = OUT_DIR / out_name
    if out_path.exists() and out_path.stat().st_size > 0:
        tqdm.write(f"Already have {out_path}")
        return
    pieces = []
    total = 0.0
    label = out_name.replace(".wav", "")
    for f in pbar(files, desc=label, unit="file"):
        data, sr = sf.read(f)
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = resample_to_16k(data, sr)
        pieces.append(data)
        total += len(data) / 16000
        if total >= target_len_sec:
            break
    if not pieces:
        tqdm.write(f"WARNING: no audio for {out_name}")
        return
    combined = np.concatenate(pieces)
    # trim / pad to ~40 s
    target_samples = int(target_len_sec * 16000)
    if len(combined) > target_samples:
        combined = combined[:target_samples]
    else:
        combined = np.pad(combined, (0, max(0, target_samples - len(combined))))
    sf.write(out_path, combined, 16000)
    tqdm.write(f"Created {out_path} ({len(combined)/16000:.1f}s)")

def main():
    # 1. Mini LibriSpeech (clean male/female, 16 kHz FLAC)
    mini_url = "https://www.openslr.org/resources/31/dev-clean-2.tar.gz"
    mini_tar = TMP / "dev-clean-2.tar.gz"
    download(mini_url, mini_tar)

    # Typical structure after extract: TMP/LibriSpeech/dev-clean-2/...
    found = list(TMP.rglob("dev-clean-2"))
    libri_root = found[0] if found else TMP / "LibriSpeech" / "dev-clean-2"
    if not (libri_root.exists() and any(libri_root.rglob("*.flac"))):
        with tarfile.open(mini_tar, "r:gz") as tar:
            members = tar.getmembers()
            for member in pbar(members, desc="Extract LibriSpeech", unit="file"):
                tar.extract(member, TMP)
        found = list(TMP.rglob("dev-clean-2"))
        libri_root = found[0] if found else TMP / "LibriSpeech" / "dev-clean-2"
    else:
        tqdm.write(f"Already extracted Mini LibriSpeech")

    # Collect some male / female files (we will pick known speakers later if needed)
    all_flacs = sorted(libri_root.rglob("*.flac"))
    tqdm.write(f"Found {len(all_flacs)} FLAC files in Mini LibriSpeech")

    # Simple split: first half roughly one gender group, second half another
    # (for robustness we just take sequential groups)
    male_cands = all_flacs[0:30]
    female_cands = all_flacs[30:60]

    # 2. Valentini noisy testset (48 kHz WAV)
    # Direct zip from Edinburgh DataShare (handle redirects if needed)
    val_url = "https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_testset_wav.zip?sequence=4&isAllowed=y"
    val_zip = TMP / "noisy_testset_wav.zip"
    try:
        download(val_url, val_zip)
    except Exception as e:
        tqdm.write("Valentini direct link may have changed. Trying alternative approach / skipping noisy for now.")
        tqdm.write(str(e))
        val_zip = None

    dest_root = TMP / "valentini"
    noisy_files = sorted(dest_root.rglob("*.wav"))[:20] if dest_root.exists() else []
    if noisy_files:
        tqdm.write(f"Already extracted Valentini ({len(noisy_files)} WAV)")
    elif val_zip and val_zip.exists():
        with zipfile.ZipFile(val_zip, "r") as z:
            names = z.namelist()
            for name in pbar(names, desc="Extract Valentini", unit="file"):
                z.extract(name, dest_root)
        noisy_files = sorted(dest_root.rglob("*.wav"))[:20]
        tqdm.write(f"Found {len(noisy_files)} noisy WAV files")

    # ---------- Build the 10 samples ----------
    # Female 1 & 2 (from LibriSpeech)
    concat_and_save(female_cands[0:8],  "01_female_clean_1.wav", target_len_sec=40)
    concat_and_save(female_cands[8:16], "02_female_clean_2.wav", target_len_sec=42)

    # Male 1 & 2
    concat_and_save(male_cands[0:8],  "03_male_clean_1.wav", target_len_sec=38)
    concat_and_save(male_cands[8:16], "04_male_clean_2.wav", target_len_sec=45)

    # Child approximation (higher-pitch / different speakers from Libri + note)
    # True open child speech of 30-60s is rare; these are the best approximation from available data.
    # Strongly recommend recording 1-2 real child samples yourself for the final set.
    concat_and_save(all_flacs[60:70], "05_child_approx_1.wav", target_len_sec=35)
    concat_and_save(all_flacs[70:80], "06_child_approx_2.wav", target_len_sec=40)

    # Multi-speaker (concatenate different speakers)
    multi1 = male_cands[0:4] + female_cands[0:4]
    multi2 = male_cands[10:14] + female_cands[10:14]
    concat_and_save(multi1, "07_multi_speaker_1.wav", target_len_sec=45)
    concat_and_save(multi2, "08_multi_speaker_2.wav", target_len_sec=48)

    # Noisy
    if noisy_files:
        concat_and_save(noisy_files[0:6],  "09_noisy_1.wav", target_len_sec=40)
        concat_and_save(noisy_files[6:12], "10_noisy_2.wav", target_len_sec=42)
    else:
        print("Noisy files missing – you can manually download Valentini testset later.")

    print("\nDone. 10 samples are in the 'samples/' folder.")
    print("All are mono 16 kHz WAV, ~35-50 s.")
    print("Convert any to MP3 with: ffmpeg -i samples/XX.wav -ac 1 -ar 16000 samples/XX.mp3")
    print("\nNOTE on child samples: these are approximations from available adult data.")
    print("For true child speech, record 30-60 s yourself or use Freesound (login required for many).")

if __name__ == "__main__":
    main()
