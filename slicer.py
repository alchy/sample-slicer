#
# python slicer.py --input-dir samples_in --output-dir samples_out_sliced --threshold_db -45 --min_length 3
#

import os
import numpy as np
import wave
import argparse
from tqdm import tqdm
import logging
import sys


# --- Konfigurace logování pro spolupráci s tqdm ---
class TqdmStreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, file=sys.stderr)
            self.flush()
        except Exception:
            self.handleError(record)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if logger.hasHandlers():
    logger.handlers.clear()

handler = TqdmStreamHandler()
formatter = logging.Formatter('%(levelname)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


# --- Konec konfigurace logování ---

def detect_segments(data, fs, threshold_db=-40, window_size=0.05, min_segment_length=3):
    """
    Detekuje segmenty zvuku nad prahem energie (RMS).
    Optimalizováno pro rychlost.
    """
    if len(data) == 0:
        return []

    max_abs = np.max(np.abs(data))
    if max_abs == 0:
        return []

    hop_length = int(window_size * fs)
    if hop_length == 0:
        hop_length = 1  # Zajisti, aby hop_length nebyl nula

    # Vektorizovaný výpočet RMS
    # Pokud data nejsou dělitelná hop_length, poslední část se zahodí
    num_windows = len(data) // hop_length
    if num_windows == 0:  # Zabrání chybě, pokud je signál příliš krátký pro jakékoli okno
        return []

    # Přetvarování dat na bloky a výpočet RMS přes osy
    reshaped_data = data[:num_windows * hop_length].reshape(-1, hop_length)
    rms = np.sqrt(np.mean(reshaped_data ** 2, axis=1))

    # Převod RMS na dB
    rms_db = 20 * np.log10(rms / (max_abs + 1e-10))

    active = rms_db > threshold_db
    segments = []
    start_window_idx = None

    for i in range(len(active)):
        if active[i] and start_window_idx is None:
            start_window_idx = i
        elif not active[i] and start_window_idx is not None:
            end_window_idx = i
            start_sample = start_window_idx * hop_length
            end_sample = end_window_idx * hop_length
            if (end_sample - start_sample) / fs >= min_segment_length:
                segments.append((start_sample, end_sample))
            start_window_idx = None

    # Zpracování posledního segmentu
    if start_window_idx is not None:
        start_sample = start_window_idx * hop_length
        end_sample = len(data)  # Poslední segment jde až do konce dat
        if (end_sample - start_sample) / fs >= min_segment_length:
            segments.append((start_sample, end_sample))

    return segments


def process_wav(input_path, output_dir, threshold_db, min_length):
    """
    Zpracuje jeden WAV soubor: detekuje segmenty, ořeže je a uloží je jako samostatné WAV soubory.
    """
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    logger.info(f"Načítám soubor: {os.path.basename(input_path)}")
    try:
        with wave.open(input_path, 'rb') as wf:
            fs = wf.getframerate()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            if sampwidth != 2:
                raise ValueError("Podporovány pouze 16-bit WAV soubory.")
            n_frames = wf.getnframes()
            raw_data = wf.readframes(n_frames)
        data = np.frombuffer(raw_data, dtype=np.int16).reshape(-1, n_channels)
    except Exception as e:
        logger.error(f"Chyba při načítání souboru {os.path.basename(input_path)}: {e}")
        return

    if data.ndim == 1:
        data = data[:, np.newaxis]

    # Převedeme data na float32 pro přesné zpracování
    # Zabrání přetečení při umocňování a umožní přesnější výpočty
    data = data.astype(np.float32)

    logger.info("Očišťuji DC offset...")
    means = np.mean(data, axis=0)
    data -= means

    mono_data = np.mean(data, axis=1)

    logger.info("Detekuji segmenty...")
    segments = detect_segments(mono_data, fs, threshold_db, min_segment_length=min_length)
    logger.info(f"Nalezeno {len(segments)} segmentů.")

    for idx, (start_orig, end_orig) in enumerate(
            tqdm(segments, desc=f"Zpracovávám segmenty z {os.path.basename(input_path)}", unit="segment", leave=False)):

        stereo_segment_orig = data[start_orig:end_orig, :]
        mono_segment = mono_data[start_orig:end_orig]

        # --- Ořezávání koncového a počátečního ticha - Vektorizovaná verze ---
        trim_threshold_db = threshold_db + 10
        trim_window_size = 0.01  # 10 ms okno
        hop_length_trim = int(trim_window_size * fs)
        if hop_length_trim == 0: hop_length_trim = 1

        if len(mono_segment) == 0:
            logger.info(f"Segment {idx + 1} je prázdný před ořezáním, přeskakuji.")
            continue

        segment_max_abs = np.max(np.abs(stereo_segment_orig))
        if segment_max_abs == 0:
            logger.debug(f"Segment {idx + 1} je celý tichý, označuji jako prázdný po ořezání.")
            mono_segment_trimmed = np.array([])
            stereo_segment_trimmed = np.array([])
        else:
            # Vektorizovaný výpočet RMS pro trimování
            num_trim_windows = len(mono_segment) // hop_length_trim
            if num_trim_windows == 0:  # Pokud je segment kratší než jedno okno pro trimování
                mono_segment_trimmed = np.array([])
                stereo_segment_trimmed = np.array([])
            else:
                # Ořezání dat na celočíselný násobek hop_length_trim pro reshape
                mono_segment_for_rms = mono_segment[:num_trim_windows * hop_length_trim]
                reshaped_mono_segment = mono_segment_for_rms.reshape(-1, hop_length_trim)
                rms_windows = np.sqrt(np.mean(reshaped_mono_segment ** 2, axis=1))
                rms_windows_db = 20 * np.log10(rms_windows / (segment_max_abs + 1e-10))

                active_windows = np.where(rms_windows_db >= trim_threshold_db)[0]

                if len(active_windows) > 0:
                    first_active_window = active_windows[0]
                    last_active_window = active_windows[-1]

                    effective_start_idx = first_active_window * hop_length_trim
                    # effective_end_idx by měla jít až na konec posledního aktivního okna
                    effective_end_idx = (last_active_window + 1) * hop_length_trim

                    # Zajisti, aby effective_end_idx nepřesáhla původní délku segmentu
                    effective_end_idx = min(effective_end_idx, len(mono_segment))

                else:
                    effective_start_idx = 0
                    effective_end_idx = 0  # Celý segment je pod prahem

                mono_segment_trimmed = mono_segment[effective_start_idx:effective_end_idx]
                stereo_segment_trimmed = stereo_segment_orig[effective_start_idx:effective_end_idx, :]

        if len(mono_segment_trimmed) == 0:
            logger.info(f"Segment {idx + 1} je po ořezání prázdný, přeskakuji.")
            continue

        if len(mono_segment_trimmed) / fs < 0.1:  # Min. délka po ořezání
            logger.info(
                f"Segment {idx + 1} je po ořezání příliš krátký ({len(mono_segment_trimmed) / fs:.3f}s), přeskakuji.")
            continue

        # Časový timestamp: start v ms
        start_ms = int(start_orig / fs * 1000)

        # Název souboru
        base_name_template = f"{base_name}_slice_{idx+1}_start_{start_ms}ms"
        output_filename = f"{base_name_template}.wav"
        output_path = os.path.join(output_dir, output_filename)

        # Zajištění unikátnosti názvu souboru
        i = 1
        while os.path.exists(output_path):
            output_filename = f"{base_name_template}_{i}.wav"
            output_path = os.path.join(output_dir, output_filename)
            i += 1

        try:
            with wave.open(output_path, 'wb') as wf:
                wf.setnchannels(stereo_segment_trimmed.shape[1])
                wf.setsampwidth(2)  # 16-bit
                wf.setframerate(fs)
                wf.writeframes(stereo_segment_trimmed.astype(np.int16).tobytes())
            logger.info(f"Uložen segment {idx + 1}: {os.path.basename(output_path)}")
        except Exception as e:
            logger.error(f"Chyba při ukládání souboru {os.path.basename(output_path)}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Program pro rozdělení WAV souborů na jednotlivé segmenty zvuku a jejich uložení. "
                    "Detekuje segmenty nad zadaným prahem hlasitosti a minimální délkou. "
                    "Ukládá segmenty s názvy obsahujícími časový timestamp počátku segmentu."
    )
    parser.add_argument("--input-dir", required=True, help="Vstupní adresář s WAV soubory")
    parser.add_argument("--output-dir", required=True, help="Výstupní adresář pro segmenty")
    parser.add_argument("--threshold_db", type=float, default=-45, help="Práh detekce v dB (výchozí: -45)")
    parser.add_argument("--min_length", type=float, default=3,
                        help="Minimální délka segmentu v sekundách (výchozí: 3)")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Úroveň logování (výchozí: INFO)")

    args = parser.parse_args()

    logger.setLevel(getattr(logging, args.log_level.upper()))

    os.makedirs(args.output_dir, exist_ok=True)

    wav_files = [f for f in os.listdir(args.input_dir) if f.endswith(".wav")]
    logger.info(f"Nalezeno {len(wav_files)} WAV souborů k zpracování.")

    # Bez paralelizace (pro jednoduchost, pokud není potřeba složitá správa procesů)
    for filename in tqdm(wav_files, desc="Zpracovávám soubory", unit="soubor"):
        input_path = os.path.join(args.input_dir, filename)
        process_wav(input_path, args.output_dir, args.threshold_db, args.min_length)

    logger.info("Zpracování dokončeno.")


if __name__ == "__main__":
    main()