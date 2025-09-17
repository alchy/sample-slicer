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
from pathlib import Path
import json


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


# --- Statistiky zpracování ---
class ProcessingStats:
    def __init__(self):
        self.files_processed = 0
        self.files_failed = 0
        self.segments_created = 0
        self.segments_skipped = 0
        self.total_input_duration = 0
        self.total_output_duration = 0
        self.audio_formats = {}

    def add_format(self, sample_rate, channels, bit_depth):
        format_key = f"{sample_rate}Hz_{channels}ch_{bit_depth}bit"
        self.audio_formats[format_key] = self.audio_formats.get(format_key, 0) + 1

    def print_summary(self):
        logger.info("=== SOUHRN ZPRACOVÁNÍ ===")
        logger.info(f"Soubory: {self.files_processed} úspěšné, {self.files_failed} neúspěšné")
        logger.info(f"Segmenty: {self.segments_created} vytvořené, {self.segments_skipped} přeskočené")
        logger.info(f"Doba: {self.total_input_duration:.1f}s vstup → {self.total_output_duration:.1f}s výstup")
        logger.info("Formáty:")
        for format_name, count in self.audio_formats.items():
            logger.info(f"  {format_name}: {count} souborů")


def validate_inputs(args):
    """Validuje vstupní parametry"""
    errors = []

    if not Path(args.input_dir).exists():
        errors.append(f"Vstupní adresář neexistuje: {args.input_dir}")

    if args.threshold_db > 0:
        errors.append(f"Práh musí být záporný nebo nulový, zadáno: {args.threshold_db}")

    if args.min_length < 0:
        errors.append(f"Minimální délka musí být kladná, zadáno: {args.min_length}")

    if args.min_length_after_trim < 0:
        errors.append(f"Minimální délka po ořezání musí být kladná, zadáno: {args.min_length_after_trim}")

    if errors:
        for error in errors:
            logger.error(error)
        return False

    return True


def apply_fade(data, sample_rate, max_fade_ms=2.5):
    """Aplikuje fade-in a fade-out pro odstranění kliku (max 5ms)"""
    # Výpočet fade samples na základě vzorkovací frekvence
    fade_samples = int(max_fade_ms * sample_rate / 1000.0)

    if len(data) <= 2 * fade_samples:
        # Pro krátké segmenty použij kratší fade (max 1/4 délky segmentu)
        fade_samples = len(data) // 4

    if fade_samples > 0:
        # Fade-in
        fade_in = np.linspace(0, 1, fade_samples)
        if data.ndim == 1:
            data[:fade_samples] *= fade_in
        else:
            data[:fade_samples] *= fade_in[:, np.newaxis]

        # Fade-out
        fade_out = np.linspace(1, 0, fade_samples)
        if data.ndim == 1:
            data[-fade_samples:] *= fade_out
        else:
            data[-fade_samples:] *= fade_out[:, np.newaxis]

    return data


def detect_segments(data, fs, threshold_db=-40, window_size=0.05, min_segment_length=3):
    """
    Detekuje segmenty zvuku nad prahem energie (RMS).
    Optimalizováno pro různé vzorkovací frekvence.
    """
    if len(data) == 0:
        return []

    # Normalizace pro výpočet dB
    max_abs = np.max(np.abs(data))
    if max_abs == 0:
        return []

    # Adaptivní velikost okna podle vzorkovací frekvence
    hop_length = max(1, int(window_size * fs))

    # Vektorizovaný výpočet RMS
    num_windows = len(data) // hop_length
    if num_windows == 0:
        return []

    # Výpočet RMS pro každé okno
    reshaped_data = data[:num_windows * hop_length].reshape(-1, hop_length)
    rms = np.sqrt(np.mean(reshaped_data ** 2, axis=1))

    # Převod RMS na dB
    rms_db = 20 * np.log10(rms / max_abs + 1e-10)

    # Detekce aktivních segmentů
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

            # Kontrola minimální délky
            if (end_sample - start_sample) / fs >= min_segment_length:
                segments.append((start_sample, end_sample))
            start_window_idx = None

    # Zpracování posledního segmentu
    if start_window_idx is not None:
        start_sample = start_window_idx * hop_length
        end_sample = len(data)
        if (end_sample - start_sample) / fs >= min_segment_length:
            segments.append((start_sample, end_sample))

    return segments


def trim_silence(data, fs, threshold_db, trim_window_size=0.01):
    """
    Ořeže ticho ze začátku a konce segmentu.
    Vrací ořezaná data nebo prázdné pole pokud je vše ticho.
    """
    if len(data) == 0:
        return np.array([]), 0, 0

    # Pro mono data
    if data.ndim == 1:
        mono_data = data
    else:
        mono_data = np.mean(data, axis=1)

    max_abs = np.max(np.abs(mono_data))
    if max_abs == 0:
        return np.array([]), 0, 0

    # Adaptivní velikost okna
    hop_length = max(1, int(trim_window_size * fs))
    num_windows = len(mono_data) // hop_length

    if num_windows == 0:
        return np.array([]), 0, 0

    # Výpočet RMS pro trimování
    reshaped_mono = mono_data[:num_windows * hop_length].reshape(-1, hop_length)
    rms_windows = np.sqrt(np.mean(reshaped_mono ** 2, axis=1))
    rms_windows_db = 20 * np.log10(rms_windows / max_abs + 1e-10)

    # Najdi aktivní okna
    active_windows = np.where(rms_windows_db >= threshold_db)[0]

    if len(active_windows) == 0:
        return np.array([]), 0, 0

    # Určí rozsah ořezání
    first_active = active_windows[0]
    last_active = active_windows[-1]

    start_idx = first_active * hop_length
    end_idx = min((last_active + 1) * hop_length, len(data))

    return data[start_idx:end_idx], start_idx, end_idx


def get_audio_info(file_path):
    """Získá informace o audio souboru"""
    try:
        with wave.open(file_path, 'rb') as wf:
            info = {
                'sample_rate': wf.getframerate(),
                'channels': wf.getnchannels(),
                'sample_width': wf.getsampwidth(),
                'frames': wf.getnframes(),
                'duration': wf.getnframes() / wf.getframerate()
            }

            # Určení bit depth
            bit_depth = info['sample_width'] * 8
            info['bit_depth'] = bit_depth

            # Určení typu (mono/stereo)
            info['type'] = 'mono' if info['channels'] == 1 else 'stereo'

            return info
    except Exception as e:
        logger.error(f"Nelze získat info o souboru {file_path}: {e}")
        return None


def load_audio_data(file_path, audio_info):
    """Načte audio data s ohledem na formát"""
    try:
        with wave.open(file_path, 'rb') as wf:
            raw_data = wf.readframes(audio_info['frames'])

        # Konverze podle bit depth
        if audio_info['bit_depth'] == 16:
            dtype = np.int16
            scale = 32767.0
        elif audio_info['bit_depth'] == 24:
            # 24-bit není přímo podporováno numpy, ale můžeme simulovat
            logger.warning(f"24-bit audio není plně podporováno: {file_path}")
            dtype = np.int16  # Fallback
            scale = 32767.0
        elif audio_info['bit_depth'] == 32:
            dtype = np.int32
            scale = 2147483647.0
        else:
            logger.error(f"Nepodporovaná bit depth {audio_info['bit_depth']}: {file_path}")
            return None

        # Načtení a reshape dat
        data = np.frombuffer(raw_data, dtype=dtype)

        if audio_info['channels'] > 1:
            data = data.reshape(-1, audio_info['channels'])

        # Konverze na float32 pro zpracování
        data_float = data.astype(np.float32) / scale

        return data_float, scale, dtype

    except Exception as e:
        logger.error(f"Chyba při načítání audio dat {file_path}: {e}")
        return None


def save_audio_segment(data, output_path, audio_info, original_scale, original_dtype):
    """Uloží audio segment se zachováním původních parametrů"""
    try:
        # Konverze zpět na původní formát
        if original_dtype == np.int16:
            data_int = (data * original_scale).clip(-32768, 32767).astype(np.int16)
        elif original_dtype == np.int32:
            data_int = (data * original_scale).clip(-2147483648, 2147483647).astype(np.int32)
        else:
            data_int = (data * original_scale).astype(original_dtype)

        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(audio_info['channels'])
            wf.setsampwidth(audio_info['sample_width'])
            wf.setframerate(audio_info['sample_rate'])
            wf.writeframes(data_int.tobytes())

        return True
    except Exception as e:
        logger.error(f"Chyba při ukládání {output_path}: {e}")
        return False


def process_wav(input_path, output_dir, threshold_db, min_length, min_length_after_trim, trim_threshold_offset,
                apply_fades, fade_ms, stats):
    """
    Zpracuje jeden WAV soubor s podporou různých formátů a vzorkovacích frekvencí.
    """
    base_name = Path(input_path).stem
    logger.info(f"Načítám soubor: {Path(input_path).name}")

    # Získání informací o souboru
    audio_info = get_audio_info(input_path)
    if not audio_info:
        stats.files_failed += 1
        return

    # Logování formátu
    logger.info(
        f"Formát: {audio_info['sample_rate']}Hz, {audio_info['channels']} kanál(y), {audio_info['bit_depth']}-bit")
    logger.info(f"Délka: {audio_info['duration']:.2f}s")

    stats.add_format(audio_info['sample_rate'], audio_info['channels'], audio_info['bit_depth'])
    stats.total_input_duration += audio_info['duration']

    # Načtení audio dat
    result = load_audio_data(input_path, audio_info)
    if not result:
        stats.files_failed += 1
        return

    data, original_scale, original_dtype = result

    # DC offset korekce
    logger.debug("Odstraňuji DC offset...")
    if data.ndim == 1:
        data -= np.mean(data)
        mono_data = data
    else:
        data -= np.mean(data, axis=0)
        mono_data = np.mean(data, axis=1)

    # Detekce segmentů
    logger.debug("Detekuji segmenty...")
    segments = detect_segments(
        mono_data,
        audio_info['sample_rate'],
        threshold_db,
        min_segment_length=min_length
    )
    logger.info(f"Nalezeno {len(segments)} segmentů.")

    if not segments:
        logger.warning("Nebyly nalezeny žádné segmenty.")
        stats.files_processed += 1
        return

    # Zpracování segmentů
    trim_threshold_db = threshold_db + trim_threshold_offset
    segments_saved = 0

    for idx, (start_sample, end_sample) in enumerate(
            tqdm(segments, desc=f"Segmenty z {Path(input_path).name}", unit="segment", leave=False)
    ):

        # Extrakce segmentu
        if data.ndim == 1:
            segment_data = data[start_sample:end_sample].copy()
        else:
            segment_data = data[start_sample:end_sample, :].copy()

        # Ořezání ticha
        trimmed_data, trim_start, trim_end = trim_silence(
            segment_data,
            audio_info['sample_rate'],
            trim_threshold_db
        )

        if len(trimmed_data) == 0:
            logger.debug(f"Segment {idx + 1} je po ořezání prázdný, přeskakuji.")
            stats.segments_skipped += 1
            continue

        # Kontrola minimální délky po ořezání
        trimmed_duration = len(trimmed_data) / audio_info['sample_rate']
        if trimmed_duration < min_length_after_trim:
            logger.debug(f"Segment {idx + 1} je po ořezání příliš krátký ({trimmed_duration:.3f}s), přeskakuji.")
            stats.segments_skipped += 1
            continue

        # Aplikace fade-in/fade-out
        if apply_fades:
            trimmed_data = apply_fade(trimmed_data, audio_info['sample_rate'], fade_ms)

        # Název výstupního souboru
        start_ms = int((start_sample + trim_start) / audio_info['sample_rate'] * 1000)
        duration_ms = int(trimmed_duration * 1000)

        # Informace o formátu v názvu souboru
        format_info = f"{audio_info['sample_rate'] // 1000}k_{audio_info['type']}"
        base_filename = f"{base_name}_slice_{idx + 1:03d}_start_{start_ms}ms_dur_{duration_ms}ms_{format_info}"

        output_filename = f"{base_filename}.wav"
        output_path = Path(output_dir) / output_filename

        # Opravená logika pro zajištění unikátnosti: Pouze pokud existuje, přidej sufix
        counter = 1
        while output_path.exists():
            output_filename = f"{base_filename}_{counter}.wav"
            output_path = Path(output_dir) / output_filename
            counter += 1

        # Uložení segmentu
        if save_audio_segment(trimmed_data, str(output_path), audio_info, original_scale, original_dtype):
            logger.debug(f"Uložen segment {idx + 1}: {output_filename} ({trimmed_duration:.2f}s)")
            segments_saved += 1
            stats.segments_created += 1
            stats.total_output_duration += trimmed_duration
        else:
            stats.segments_skipped += 1

    logger.info(f"Uloženo {segments_saved}/{len(segments)} segmentů z {Path(input_path).name}")
    stats.files_processed += 1


def main():
    parser = argparse.ArgumentParser(
        description="Vylepšený program pro rozdělení WAV souborů na segmenty. "
                    "Podporuje různé vzorkovací frekvence (44.1kHz, 48kHz, atd.) "
                    "a automaticky detekuje mono/stereo formát. "
                    "Zachovává původní audio parametry ve výstupních souborech."
    )
    parser.add_argument("--input-dir", required=True, help="Vstupní adresář s WAV soubory")
    parser.add_argument("--output-dir", required=True, help="Výstupní adresář pro segmenty")
    parser.add_argument("--threshold_db", type=float, default=-45, help="Práh detekce v dB (výchozí: -45)")
    parser.add_argument("--min_length", type=float, default=3.0,
                        help="Minimální délka segmentu v sekundách (výchozí: 3.0)")
    parser.add_argument("--min_length_after_trim", type=float, default=0.5,
                        help="Minimální délka segmentu po ořezání ticha v sekundách (výchozí: 0.5)")
    parser.add_argument("--trim_threshold_offset", type=float, default=10,
                        help="Offset pro práh ořezávání vzhledem k detekčnímu prahu v dB (výchozí: +10)")
    parser.add_argument("--fade_ms", type=float, default=5.0,
                        help="Délka fade-in/fade-out v milisekundách (výchozí: 5.0)")
    parser.add_argument("--no_fades", action="store_true",
                        help="Zakáže fade-in/fade-out (může způsobit kliknutí)")
    parser.add_argument("--resume", action="store_true",
                        help="Přeskočí již existující výstupní soubory")
    parser.add_argument("--preview", action="store_true",
                        help="Preview mód - zobrazí co by se stalo bez skutečného zpracování")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Úroveň logování (výchozí: INFO)")

    args = parser.parse_args()

    # Nastavení logování
    logger.setLevel(getattr(logging, args.log_level.upper()))

    # Validace vstupů
    if not validate_inputs(args):
        return 1

    # Vytvoření výstupního adresáře
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Nalezení WAV souborů
    input_path = Path(args.input_dir)
    wav_files = list(input_path.glob("*.wav")) + list(input_path.glob("*.WAV"))

    if not wav_files:
        logger.error(f"V adresáři {args.input_dir} nebyly nalezeny žádné WAV soubory.")
        return 1

    logger.info(f"Nalezeno {len(wav_files)} WAV souborů k zpracování.")

    if args.preview:
        logger.info("=== PREVIEW MÓD - žádné soubory nebudou změněny ===")
        for wav_file in wav_files:
            audio_info = get_audio_info(str(wav_file))
            if audio_info:
                logger.info(f"{wav_file.name}: {audio_info['sample_rate']}Hz, "
                            f"{audio_info['channels']}ch, {audio_info['bit_depth']}-bit, "
                            f"{audio_info['duration']:.2f}s")
        return 0

    # Inicializace statistik
    stats = ProcessingStats()

    # Zpracování souborů
    apply_fades = not args.no_fades

    for wav_file in tqdm(wav_files, desc="Zpracovávám soubory", unit="soubor"):
        if args.resume:
            # Jednoduché resume - zkontroluje, zda existují nějaké výstupy pro tento soubor
            base_name = wav_file.stem
            existing_outputs = list(Path(args.output_dir).glob(f"{base_name}_slice_*.wav"))
            if existing_outputs:
                logger.info(f"Přeskakuji {wav_file.name} - nalezeny existující výstupy")
                continue

        process_wav(
            str(wav_file),
            args.output_dir,
            args.threshold_db,
            args.min_length,
            args.min_length_after_trim,
            args.trim_threshold_offset,
            apply_fades,
            args.fade_ms,
            stats
        )

    # Výpis souhrnných statistik
    stats.print_summary()
    logger.info("Zpracování dokončeno.")

    return 0


if __name__ == "__main__":
    exit(main())