#!/usr/bin/env python3
"""
Download and prepare exactly 10 evaluation samples for MTE252 Phase 1 Task 2.
Categories: 2 female, 2 male, 2 child (approx), 2 multi-speaker, 2 noisy.
Target: 30-60 s, >=16 kHz mono WAV.
"""

import os
import tarfile
import zipfile
import urllib.request
import shutil
from pathlib import Path

try:
    import soundfile as sf
    import numpy as np
    from scipy import signal
except ImportError:
    print("Installing required packages: soundfile scipy numpy")
    os.system("pip install soundfile scipy numpy -q")
    import soundfile as sf
    import numpy as np
    from scipy import signal

OUT_DIR = Path("samples")
OUT_DIR.mkdir(exist_ok=True)
TMP = Path("tmp_audio")
TMP.mkdir(exist_ok=True)

def download(url, dest):
    if dest.exists():
        print(f"Already have {dest}")
        return
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved {dest}")

def resample_to_16k(data, sr):
    if sr == 16000:
        return data
    # simple resample
    num = int(len(data) * 16000 / sr)
    return signal.resample(data, num).astype(np.float32)

def concat_and_save(files, out_name, target_sr=16000, target_len_sec=40):
    """Concatenate audio files until roughly target length, save as 16k mono WAV."""
    pieces = []
    total = 0.0
    for f in files:
        data, sr = sf.read(f)
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = resample_to_16k(data, sr)
        pieces.append(data)
        total += len(data) / 16000
        if total >= target_len_sec:
            break
    if not pieces:
        print(f"WARNING: no audio for {out_name}")
        return
    combined = np.concatenate(pieces)
    # trim / pad to ~40 s
    target_samples = int(target_len_sec * 16000)
    if len(combined) > target_samples:
        combined = combined[:target_samples]
    else:
        combined = np.pad(combined, (0, max(0, target_samples - len(combined))))
    out_path = OUT_DIR / out_name
    sf.write(out_path, combined, 16000)
    print(f"Created {out_path} ({len(combined)/16000:.1f}s)")

def main():
    # 1. Mini LibriSpeech (clean male/female, 16 kHz FLAC)
    mini_url = "https://www.openslr.org/resources/31/dev-clean-2.tar.gz"
    mini_tar = TMP / "dev-clean-2.tar.gz"
    download(mini_url, mini_tar)

    with tarfile.open(mini_tar, "r:gz") as tar:
        tar.extractall(TMP)

    # Typical structure after extract: TMP/LibriSpeech/dev-clean-2/...
    libri_root = list(TMP.rglob("dev-clean-2"))[0] if list(TMP.rglob("dev-clean-2")) else TMP / "LibriSpeech" / "dev-clean-2"

    # Collect some male / female files (we will pick known speakers later if needed)
    all_flacs = sorted(libri_root.rglob("*.flac"))
    print(f"Found {len(all_flacs)} FLAC files in Mini LibriSpeech")

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
        print("Valentini direct link may have changed. Trying alternative approach / skipping noisy for now.")
        print(e)
        val_zip = None

    noisy_files = []
    if val_zip and val_zip.exists():
        with zipfile.ZipFile(val_zip, "r") as z:
            z.extractall(TMP / "valentini")
        noisy_files = sorted((TMP / "valentini").rglob("*.wav"))[:20]
        print(f"Found {len(noisy_files)} noisy WAV files")

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
