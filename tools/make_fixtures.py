#!/usr/bin/env python3
"""Vyřeže testovací fixtures pro pitch z reálných nahrávek.

Použití: tools/make_fixtures.py <raw-dir> <truth.json> <out-dir>
truth.json: {"260917_0180.wav": {"start": "A0", "pattern": "chromatic"}, ...}
Pro každý soubor: detekce úderů, výpis počtu přijatých úderů (ručně zkontrolovat
proti očekávání), zápis mono 24 bit od nasazení (1,5 s pod C3 kvůli hlasům k=8, jinak 0,6 s) + truth.json s MIDI.
"""
import json, sys
from pathlib import Path
import numpy as np
from sample_slicer.io import read_wav, write_wav, remove_dc, to_mono
from sample_slicer.detect import detect_segments
from sample_slicer.notes import expand_truth, midi_to_name

raw, truth_path, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
truth = json.loads(truth_path.read_text())
result = {}
for name, entry in truth.items():
    data, info = read_wav(raw / name)
    mono = to_mono(remove_dc(data))
    accepted, rejected = detect_segments(mono, info.sample_rate)
    midis = expand_truth(entry, len(accepted))
    print(f"{name}: {len(accepted)} úderů ({len(rejected)} kliků) → {midi_to_name(midis[0])}..{midi_to_name(midis[-1])}")
    for seg in accepted:
        print(f"   @{seg.onset/info.sample_rate:8.2f}s  len {(seg.end-seg.onset)/info.sample_rate:5.1f}s  peak {seg.peak_db:6.1f}  end {seg.end_reason}")
    for i, (seg, midi) in enumerate(zip(accepted, midis)):
        clip_s = 1.5 if midi < 48 else 0.6
        clip = mono[seg.onset: seg.onset + int(clip_s * info.sample_rate)].astype(np.float32)[:, None]
        fn = f"{Path(name).stem}_{i:02d}.wav"
        write_wav(out / fn, clip, info.sample_rate, 24)
        result[fn] = midi
(out / "truth.json").write_text(json.dumps(result, indent=1))
print(f"zapsáno {len(result)} fixtures do {out}")
