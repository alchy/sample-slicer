import os
import sys
import wave
import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm


# --- Logging compatible s tqdm ---
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
        key = f"{sample_rate}Hz_{channels}ch_{bit_depth}bit"
        self.audio_formats[key] = self.audio_formats.get(key, 0) + 1

    def print_summary(self):
        logger.info("=== SOUHRN ZPRACOVÁNÍ ===")
        logger.info(f"Soubory: {self.files_processed} úspěšné, {self.files_failed} neúspěšné")
        logger.info(f"Segmenty: {self.segments_created} vytvořené, {self.segments_skipped} přeskočené")
        logger.info(f"Doba: {self.total_input_duration:.1f}s vstup → {self.total_output_duration:.1f}s výstup")
        logger.info("Formáty:")
        for k, v in self.audio_formats.items():
            logger.info(f"  {k}: {v} souborů")


# --- Pomocné funkce ---
def validate_inputs(args):
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
        for e in errors:
            logger.error(e)
        return False
    return True


def apply_fade(data, sample_rate, max_fade_ms=2.5):
    fade_samples = int(max_fade_ms * sample_rate / 1000.0)
    if len(data) <= 2 * fade_samples:
        fade_samples = len(data) // 4
    if fade_samples > 0:
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)
        if data.ndim == 1:
            data[:fade_samples] *= fade_in
            data[-fade_samples:] *= fade_out
        else:
            data[:fade_samples] *= fade_in[:, np.newaxis]
            data[-fade_samples:] *= fade_out[:, np.newaxis]
    return data


def detect_segments(data, fs, threshold_db=-40, window_size=0.05, min_segment_length=3):
    if len(data) == 0: return []
    max_abs = np.max(np.abs(data))
    if max_abs == 0: return []
    hop_length = max(1, int(window_size * fs))
    num_windows = len(data) // hop_length
    if num_windows == 0: return []
    reshaped_data = data[:num_windows * hop_length].reshape(-1, hop_length)
    rms = np.sqrt(np.mean(reshaped_data ** 2, axis=1))
    rms_db = 20 * np.log10(rms / max_abs + 1e-10)
    active = rms_db > threshold_db
    segments = []
    start_idx = None
    for i in range(len(active)):
        if active[i] and start_idx is None:
            start_idx = i
        elif not active[i] and start_idx is not None:
            end_idx = i
            start_sample = start_idx * hop_length
            end_sample = end_idx * hop_length
            if (end_sample - start_sample) / fs >= min_segment_length:
                segments.append((start_sample, end_sample))
            start_idx = None
    if start_idx is not None:
        start_sample = start_idx * hop_length
        end_sample = len(data)
        if (end_sample - start_sample) / fs >= min_segment_length:
            segments.append((start_sample, end_sample))
    return segments


def trim_silence(data, fs, threshold_db, trim_window_size=0.01):
    if len(data) == 0: return np.array([]), 0, 0
    mono = data if data.ndim == 1 else np.mean(data, axis=1)
    max_abs = np.max(np.abs(mono))
    if max_abs == 0: return np.array([]), 0, 0
    hop_length = max(1, int(trim_window_size * fs))
    num_windows = len(mono) // hop_length
    if num_windows == 0: return np.array([]), 0, 0
    reshaped = mono[:num_windows * hop_length].reshape(-1, hop_length)
    rms = np.sqrt(np.mean(reshaped ** 2, axis=1))
    rms_db = 20 * np.log10(rms / max_abs + 1e-10)
    active = np.where(rms_db >= threshold_db)[0]
    if len(active) == 0: return np.array([]), 0, 0
    start_idx = active[0] * hop_length
    end_idx = min((active[-1] + 1) * hop_length, len(data))
    return data[start_idx:end_idx], start_idx, end_idx


def fine_trim_start(data, fs, pre_roll_ms=10, trigger_ratio=5.0, fine_window_ms=2.0,
                    overlap=0.5, post_trigger_shift_ms=5):
    """
    Fine trimming začátku segmentu podle náběhu energie s posunem a zero-crossing.

    - pre_roll_ms: ms, kolik zachovat před prvním náběhem
    - trigger_ratio: RMS musí být N× vyšší než RMS ticha
    - fine_window_ms: velikost okna pro RMS v ms
    - overlap: překrytí mezi okny
    - post_trigger_shift_ms: posun okna po detekci náběhu do silnějšího signálu
    """
    if len(data) == 0:
        return data, 0

    # Mono pro analýzu
    mono = np.mean(data, axis=1) if data.ndim > 1 else data

    # Výpočty vzorků
    pre_roll_samples = max(1, int(fs * pre_roll_ms / 1000.0))
    fine_window_samples = max(1, int(fs * fine_window_ms / 1000.0))
    step_samples = max(1, int(fine_window_samples * (1 - overlap)))
    post_trigger_shift_samples = int(fs * post_trigger_shift_ms / 1000.0)

    # RMS ticha
    noise_rms = np.sqrt(np.mean(mono[:pre_roll_samples]**2))
    if noise_rms == 0:
        noise_rms = 1e-10

    # Hledání náběhu
    trim_start = 0
    for start in range(0, len(mono) - fine_window_samples + 1, step_samples):
        window = mono[start:start + fine_window_samples]
        rms = np.sqrt(np.mean(window**2))
        if rms > noise_rms * trigger_ratio:
            # trigger detekován
            trim_start = start + post_trigger_shift_samples
            break

    # Pokud nenalezen žádný náběh → nic neřež
    if trim_start == 0:
        return data, 0

    # Hledání nejbližšího zero-crossing před trim_start
    zero_crossings = np.where(np.diff(np.sign(mono[:trim_start])))[0]
    if len(zero_crossings) > 0:
        trim_start = zero_crossings[-1] + 1  # poslední před triggerem

    trim_start = min(trim_start, len(data))
    return data[trim_start:], trim_start


def fine_trim_start_orig(data, fs, pre_roll_ms=10, trigger_ratio=5.0, fine_window_ms=2.0, overlap=0.5):
    """
    Fine trimming začátku segmentu podle náběhu energie.

    - pre_roll_ms: kolik ms zachovat před prvním náběhem
    - trigger_ratio: RMS musí být N× vyšší než průměrná RMS ticha
    - fine_window_ms: velikost okna pro RMS v ms
    - overlap: překrytí mezi okny (0–1)

    Vrací: (trimmed_data, num_samples_trimmed)
    """
    if len(data) == 0:
        return data, 0

    # Mono pro analýzu
    mono = np.mean(data, axis=1) if data.ndim > 1 else data

    # Počet vzorků pro pre-roll (počáteční ticho)
    pre_roll_samples = max(1, int(fs * pre_roll_ms / 1000.0))
    fine_window_samples = max(1, int(fs * fine_window_ms / 1000.0))
    step_samples = max(1, int(fine_window_samples * (1 - overlap)))

    # Výpočet průměrné RMS ticha na začátku
    noise_rms = np.sqrt(np.mean(mono[:pre_roll_samples] ** 2))
    if noise_rms == 0:
        noise_rms = 1e-10  # aby nedošlo k dělení nulou

    # Klouzavé okno po segmentu
    for start in range(0, len(mono) - fine_window_samples + 1, step_samples):
        window = mono[start:start + fine_window_samples]
        rms = np.sqrt(np.mean(window ** 2))
        if rms > noise_rms * trigger_ratio:
            # nalezen prudký náběh, přidej pre-roll
            trim_start = max(0, start - pre_roll_samples)
            return data[trim_start:], trim_start

    # žádný náběh nalezen → nic neřež
    return data, 0


# --- Audio utility ---
def get_audio_info(file_path):
    try:
        with wave.open(file_path, 'rb') as wf:
            info = {
                'sample_rate': wf.getframerate(),
                'channels': wf.getnchannels(),
                'sample_width': wf.getsampwidth(),
                'frames': wf.getnframes(),
                'duration': wf.getnframes() / wf.getframerate()
            }
            info['bit_depth'] = info['sample_width'] * 8
            info['type'] = 'mono' if info['channels'] == 1 else 'stereo'
            return info
    except Exception as e:
        logger.error(f"Nelze získat info o souboru {file_path}: {e}")
        return None


def load_audio_data(file_path, audio_info):
    try:
        with wave.open(file_path, 'rb') as wf:
            raw_data = wf.readframes(audio_info['frames'])
        if audio_info['bit_depth'] == 16:
            dtype, scale = np.int16, 32767.0
        elif audio_info['bit_depth'] == 24:
            dtype, scale = np.int16, 32767.0
            logger.warning(f"24-bit fallback na 16-bit: {file_path}")
        elif audio_info['bit_depth'] == 32:
            dtype, scale = np.int32, 2147483647.0
        else:
            logger.error(f"Nepodporovaná bit depth {audio_info['bit_depth']}")
            return None
        data = np.frombuffer(raw_data, dtype=dtype)
        if audio_info['channels'] > 1:
            data = data.reshape(-1, audio_info['channels'])
        return data.astype(np.float32) / scale, scale, dtype
    except Exception as e:
        logger.error(f"Chyba při načítání audio dat {file_path}: {e}")
        return None


def save_audio_segment(data, output_path, audio_info, original_scale, original_dtype):
    try:
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


# --- Hlavní zpracování WAV ---
def process_wav(input_path, output_dir, threshold_db, min_length, min_length_after_trim,
                trim_threshold_offset, apply_fades, fade_ms, stats):
    base_name = Path(input_path).stem
    logger.info(f"Načítám soubor: {Path(input_path).name}")
    audio_info = get_audio_info(input_path)
    if not audio_info:
        stats.files_failed += 1
        return
    logger.info(
        f"Formát: {audio_info['sample_rate']}Hz, {audio_info['channels']} kanál(y), {audio_info['bit_depth']}-bit")
    logger.info(f"Délka: {audio_info['duration']:.2f}s")
    stats.add_format(audio_info['sample_rate'], audio_info['channels'], audio_info['bit_depth'])
    stats.total_input_duration += audio_info['duration']

    result = load_audio_data(input_path, audio_info)
    if not result:
        stats.files_failed += 1
        return
    data, original_scale, original_dtype = result

    # DC offset
    if data.ndim == 1:
        data -= np.mean(data)
        mono_data = data
    else:
        data -= np.mean(data, axis=0)
        mono_data = np.mean(data, axis=1)

    # Segmenty
    segments = detect_segments(mono_data, audio_info['sample_rate'], threshold_db, min_segment_length=min_length)
    logger.info(f"Nalezeno {len(segments)} segmentů.")
    if not segments:
        logger.warning("Nebyly nalezeny žádné segmenty.")
        stats.files_processed += 1
        return

    trim_threshold_db = threshold_db + trim_threshold_offset
    segments_saved = 0
    for idx, (start_sample, end_sample) in enumerate(
            tqdm(segments, desc=f"Segmenty z {Path(input_path).name}", unit="segment", leave=False)):
        segment_data = data[start_sample:end_sample].copy() if data.ndim == 1 else data[start_sample:end_sample,
                                                                                   :].copy()
        trimmed_data, trim_start, trim_end = trim_silence(segment_data, audio_info['sample_rate'], trim_threshold_db)
        if len(trimmed_data) == 0:
            stats.segments_skipped += 1
            continue

        # --- Fine trim podle výrazného náběhu ---
        fine_trimmed_data, trim_offset = fine_trim_start(trimmed_data, audio_info['sample_rate'])
        if len(fine_trimmed_data) == 0:
            stats.segments_skipped += 1
            continue
        trimmed_data = fine_trimmed_data
        trim_start += trim_offset

        # Kontrola minimální délky
        trimmed_duration = len(trimmed_data) / audio_info['sample_rate']
        if trimmed_duration < min_length_after_trim:
            stats.segments_skipped += 1
            continue

        # Fade
        if apply_fades:
            trimmed_data = apply_fade(trimmed_data, audio_info['sample_rate'], fade_ms)

        start_ms = int((start_sample + trim_start) / audio_info['sample_rate'] * 1000)
        duration_ms = int(trimmed_duration * 1000)
        format_info = f"{audio_info['sample_rate'] // 1000}k_{audio_info['type']}"
        base_filename = f"{base_name}_slice_{idx + 1:03d}_start_{start_ms}ms_dur_{duration_ms}ms_{format_info}"
        output_path = Path(output_dir) / f"{base_filename}.wav"
        counter = 1
        while output_path.exists():
            output_path = Path(output_dir) / f"{base_filename}_{counter}.wav"
            counter += 1

        if save_audio_segment(trimmed_data, str(output_path), audio_info, original_scale, original_dtype):
            segments_saved += 1
            stats.segments_created += 1
            stats.total_output_duration += trimmed_duration
        else:
            stats.segments_skipped += 1

    logger.info(f"Uloženo {segments_saved}/{len(segments)} segmentů z {Path(input_path).name}")
    stats.files_processed += 1


# --- Main ---
def main():
    parser = argparse.ArgumentParser(description="Rozdělení WAV souborů na segmenty s automatickým fine-trimem")
    parser.add_argument("--input-dir", required=True, help="Vstupní adresář s WAV soubory")
    parser.add_argument("--output-dir", required=True, help="Výstupní adresář pro segmenty")
    parser.add_argument("--threshold_db", type=float, default=-45, help="Práh detekce v dB")
    parser.add_argument("--min_length", type=float, default=3.0, help="Minimální délka segmentu")
    parser.add_argument("--min_length_after_trim", type=float, default=0.5, help="Min délka po trimu")
    parser.add_argument("--trim_threshold_offset", type=float, default=10, help="Offset pro trim threshold")
    parser.add_argument("--fade_ms", type=float, default=5.0, help="Délka fade-in/out v ms")
    parser.add_argument("--no_fades", action="store_true", help="Zakáže fade")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--resume", action="store_true", help="Přeskočí existující výstupy")
    parser.add_argument("--preview", action="store_true", help="Zobrazí info bez zpracování")

    args = parser.parse_args()
    logger.setLevel(getattr(logging, args.log_level.upper()))
    if not validate_inputs(args): return 1
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    wav_files = list(Path(args.input_dir).glob("*.[wW][aA][vV]"))
    if not wav_files:
        logger.error(f"Žádné WAV soubory v {args.input_dir}")
        return 1
    logger.info(f"Nalezeno {len(wav_files)} WAV souborů.")

    if args.preview:
        logger.info("=== PREVIEW ===")
        for wav_file in wav_files:
            info = get_audio_info(str(wav_file))
            if info:
                logger.info(
                    f"{wav_file.name}: {info['sample_rate']}Hz, {info['channels']}ch, {info['bit_depth']}bit, {info['duration']:.2f}s")
        return 0

    stats = ProcessingStats()
    apply_fades = not args.no_fades

    for wav_file in tqdm(wav_files, desc="Zpracovávám soubory", unit="soubor"):
        if args.resume:
            base_name = wav_file.stem
            existing_outputs = list(Path(args.output_dir).glob(f"{base_name}_slice_*.wav"))
            if existing_outputs:
                logger.info(f"Přeskakuji {wav_file.name} - existují výstupy")
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

    stats.print_summary()

if __name__ == "__main__":
    exit(main())

