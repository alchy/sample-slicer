"""
Audio Processing Logic
======================
Core funkce pro detekci a zpracování audio segmentů.
Převzato ze slicer.py - bez CLI specifické logiky.
"""

import numpy as np
import wave
import logging
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

logger = logging.getLogger(__name__)


# --- Statistiky zpracování ---
class ProcessingStats:
    """Statistiky pro sledování zpracování"""

    def __init__(self):
        self.files_processed = 0
        self.files_failed = 0
        self.segments_created = 0
        self.segments_skipped = 0
        self.total_input_duration = 0.0
        self.total_output_duration = 0.0
        self.audio_formats = {}

    def add_format(self, sample_rate: int, channels: int, bit_depth: int):
        """Zaznamenání audio formátu"""
        format_key = f"{sample_rate}Hz_{channels}ch_{bit_depth}bit"
        self.audio_formats[format_key] = self.audio_formats.get(format_key, 0) + 1

    def to_dict(self) -> Dict[str, Any]:
        """Export statistik jako dict"""
        return {
            "files_processed": self.files_processed,
            "files_failed": self.files_failed,
            "segments_created": self.segments_created,
            "segments_skipped": self.segments_skipped,
            "total_input_duration": self.total_input_duration,
            "total_output_duration": self.total_output_duration,
            "audio_formats": self.audio_formats
        }


# --- Validace ---
def validate_parameters(threshold_db: float, min_length: float, min_length_after_trim: float) -> List[str]:
    """
    Validuje zpracovací parametry.

    Returns:
        List chybových hlášek (prázdný = OK)
    """
    errors = []

    if threshold_db > 0:
        errors.append(f"Threshold must be negative or zero, got: {threshold_db}")

    if min_length < 0:
        errors.append(f"Min length must be positive, got: {min_length}")

    if min_length_after_trim < 0:
        errors.append(f"Min length after trim must be positive, got: {min_length_after_trim}")

    return errors


# --- Fade aplikace ---
def apply_fade(
    data: np.ndarray,
    sample_rate: int,
    fade_in_ms: float = 5.0,
    fade_out_percent: float = 20.0
) -> np.ndarray:
    """
    Aplikuje nezávislé fade-in a fade-out pro odstranění kliku.

    Args:
        data: Audio data (1D pro mono, 2D pro stereo)
        sample_rate: Vzorkovací frekvence
        fade_in_ms: Délka fade-in v milisekundách
        fade_out_percent: Fade-out délka v % z celkové délky segmentu (0-100)

    Returns:
        Data s aplikovaným fade
    """
    segment_length = len(data)

    # --- FADE-IN (v milisekundách) ---
    fade_in_samples = int(fade_in_ms * sample_rate / 1000.0)

    # Ochrana proti příliš dlouhému fade-in (max 25% délky segmentu)
    max_fade_in = segment_length // 4
    fade_in_samples = min(fade_in_samples, max_fade_in)

    if fade_in_samples > 0:
        fade_in_curve = np.linspace(0, 1, fade_in_samples)
        if data.ndim == 1:
            data[:fade_in_samples] *= fade_in_curve
        else:
            data[:fade_in_samples] *= fade_in_curve[:, np.newaxis]

    # --- FADE-OUT (v procentech délky segmentu) ---
    # Převod procent na počet vzorků
    fade_out_samples = int(segment_length * (fade_out_percent / 100.0))

    # Ochrana proti překryvu s fade-in
    max_fade_out = segment_length - fade_in_samples
    fade_out_samples = min(fade_out_samples, max_fade_out)

    if fade_out_samples > 0:
        fade_out_curve = np.linspace(1, 0, fade_out_samples)
        if data.ndim == 1:
            data[-fade_out_samples:] *= fade_out_curve
        else:
            data[-fade_out_samples:] *= fade_out_curve[:, np.newaxis]

    return data


# --- Detekce segmentů ---
def detect_segments(
    data: np.ndarray,
    fs: int,
    threshold_db: float = -40,
    window_size: float = 0.05,
    min_segment_length: float = 3
) -> List[Tuple[int, int]]:
    """
    Detekuje segmenty zvuku nad prahem energie (RMS).

    Args:
        data: Audio data (mono)
        fs: Vzorkovací frekvence
        threshold_db: Práh v dB
        window_size: Velikost okna v sekundách
        min_segment_length: Minimální délka segmentu v sekundách

    Returns:
        List tuplů (start_sample, end_sample)
    """
    if len(data) == 0:
        return []

    max_abs = np.max(np.abs(data))
    if max_abs == 0:
        return []

    # Adaptivní velikost okna
    hop_length = max(1, int(window_size * fs))
    num_windows = len(data) // hop_length

    if num_windows == 0:
        return []

    # Výpočet RMS pro každé okno
    reshaped_data = data[:num_windows * hop_length].reshape(-1, hop_length)
    rms = np.sqrt(np.mean(reshaped_data ** 2, axis=1))

    # Převod na dB
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


# --- Trimování ticha ---
def trim_silence(
    data: np.ndarray,
    fs: int,
    threshold_db: float,
    trim_window_size: float = 0.01
) -> Tuple[np.ndarray, int, int]:
    """
    Ořeže ticho ze začátku a konce segmentu.

    Args:
        data: Audio data
        fs: Vzorkovací frekvence
        threshold_db: Práh pro trimování
        trim_window_size: Velikost okna v sekundách

    Returns:
        (trimmed_data, start_idx, end_idx)
    """
    if len(data) == 0:
        return np.array([]), 0, 0

    # Pro mono/stereo
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

    # Určení rozsahu ořezání
    first_active = active_windows[0]
    last_active = active_windows[-1]

    start_idx = first_active * hop_length
    end_idx = min((last_active + 1) * hop_length, len(data))

    return data[start_idx:end_idx], start_idx, end_idx


# --- Audio I/O ---
def get_audio_info(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Získá informace o audio souboru.

    Returns:
        Dict s info nebo None při chybě
    """
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
        logger.error(f"Cannot get audio info for {file_path}: {e}")
        return None


def load_audio_data(
    file_path: str,
    audio_info: Dict[str, Any]
) -> Optional[Tuple[np.ndarray, float, np.dtype]]:
    """
    Načte audio data s ohledem na formát.

    Returns:
        (data_float32, scale, original_dtype) nebo None
    """
    try:
        with wave.open(file_path, 'rb') as wf:
            raw_data = wf.readframes(audio_info['frames'])

        # Konverze podle bit depth
        if audio_info['bit_depth'] == 16:
            dtype = np.int16
            scale = 32767.0
        elif audio_info['bit_depth'] == 24:
            logger.warning(f"24-bit audio not fully supported: {file_path}")
            dtype = np.int16
            scale = 32767.0
        elif audio_info['bit_depth'] == 32:
            dtype = np.int32
            scale = 2147483647.0
        else:
            logger.error(f"Unsupported bit depth {audio_info['bit_depth']}: {file_path}")
            return None

        # Načtení a reshape
        data = np.frombuffer(raw_data, dtype=dtype)

        if audio_info['channels'] > 1:
            data = data.reshape(-1, audio_info['channels'])

        # Konverze na float32
        data_float = data.astype(np.float32) / scale

        return data_float, scale, dtype

    except Exception as e:
        logger.error(f"Error loading audio data {file_path}: {e}")
        return None


def save_audio_segment(
    data: np.ndarray,
    output_path: str,
    audio_info: Dict[str, Any],
    original_scale: float,
    original_dtype: np.dtype
) -> bool:
    """
    Uloží audio segment se zachováním původních parametrů.

    Returns:
        True pokud úspěch
    """
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
        logger.error(f"Error saving {output_path}: {e}")
        return False


# --- Zpracování jednoho souboru ---
def process_wav_file(
    input_path: str,
    output_dir: str,
    threshold_db: float,
    min_length: float,
    min_length_after_trim: float,
    trim_threshold_offset: float,
    apply_fades: bool,
    fade_in_ms: float,
    fade_out_percent: float,
    overwrite: bool,
    stats: ProcessingStats,
    progress_callback=None,
    log_callback=None
) -> bool:
    """
    Zpracuje jeden WAV soubor.

    Args:
        progress_callback: Callable(percent: float) pro update progressu
        log_callback: Callable(message: str, level: str) pro logy

    Returns:
        True pokud úspěch
    """
    def log(msg: str, level: str = "INFO"):
        """Helper pro logování"""
        if log_callback:
            log_callback(msg, level)
        else:
            getattr(logger, level.lower())(msg)

    base_name = Path(input_path).stem
    log(f"Loading file: {Path(input_path).name}")

    # Získání info o souboru
    audio_info = get_audio_info(input_path)
    if not audio_info:
        stats.files_failed += 1
        return False

    log(f"Format: {audio_info['sample_rate']}Hz, {audio_info['channels']}ch, {audio_info['bit_depth']}-bit")
    log(f"Duration: {audio_info['duration']:.2f}s")

    stats.add_format(audio_info['sample_rate'], audio_info['channels'], audio_info['bit_depth'])
    stats.total_input_duration += audio_info['duration']

    # Načtení audio dat
    result = load_audio_data(input_path, audio_info)
    if not result:
        stats.files_failed += 1
        return False

    data, original_scale, original_dtype = result

    # DC offset korekce
    if data.ndim == 1:
        data -= np.mean(data)
        mono_data = data
    else:
        data -= np.mean(data, axis=0)
        mono_data = np.mean(data, axis=1)

    # Detekce segmentů
    log("Detecting segments...")
    segments = detect_segments(
        mono_data,
        audio_info['sample_rate'],
        threshold_db,
        min_segment_length=min_length
    )
    log(f"Found {len(segments)} segments")

    if not segments:
        log("No segments found", "WARNING")
        stats.files_processed += 1
        return True

    # Zpracování segmentů
    trim_threshold_db = threshold_db + trim_threshold_offset
    segments_saved = 0
    total_segments = len(segments)

    for idx, (start_sample, end_sample) in enumerate(segments):
        # Update progress
        if progress_callback:
            progress = (idx / total_segments) * 100
            progress_callback(progress)

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
            stats.segments_skipped += 1
            continue

        # Kontrola minimální délky
        trimmed_duration = len(trimmed_data) / audio_info['sample_rate']
        if trimmed_duration < min_length_after_trim:
            stats.segments_skipped += 1
            continue

        # Aplikace fade
        if apply_fades:
            trimmed_data = apply_fade(trimmed_data, audio_info['sample_rate'], fade_in_ms, fade_out_percent)

        # Název výstupního souboru
        start_ms = int((start_sample + trim_start) / audio_info['sample_rate'] * 1000)
        duration_ms = int(trimmed_duration * 1000)
        format_info = f"{audio_info['sample_rate'] // 1000}k_{audio_info['type']}"
        base_filename = f"{base_name}_slice_{idx + 1:03d}_start_{start_ms}ms_dur_{duration_ms}ms_{format_info}"

        output_filename = f"{base_filename}.wav"
        output_path = Path(output_dir) / output_filename

        # Zajištění unikátnosti (pouze pokud overwrite=False)
        if not overwrite:
            counter = 1
            while output_path.exists():
                output_filename = f"{base_filename}_{counter}.wav"
                output_path = Path(output_dir) / output_filename
                counter += 1

        # Uložení
        if save_audio_segment(trimmed_data, str(output_path), audio_info, original_scale, original_dtype):
            segments_saved += 1
            stats.segments_created += 1
            stats.total_output_duration += trimmed_duration
        else:
            stats.segments_skipped += 1

    # Finální progress
    if progress_callback:
        progress_callback(100.0)

    log(f"Saved {segments_saved}/{len(segments)} segments")
    stats.files_processed += 1
    return True
