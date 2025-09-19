##
## verze trim OK
##
## fine_threshold = 0.25 | 0.1


import os
import sys
import wave
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np

# --- Logging kompatibilní s tqdm ---
class TqdmStreamHandler(logging.StreamHandler):
    """Custom handler pro logování s tqdm progresbarem."""
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
    """Sbírá informace o zpracovaných souborech a segmentech."""
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
        for fmt, count in self.audio_formats.items():
            logger.info(f"  {fmt}: {count} souborů")

# --- Validace vstupních parametrů ---
def validate_inputs(args):
    errors = []
    if not Path(args.input_dir).exists():
        errors.append(f"Vstupní adresář neexistuje: {args.input_dir}")
    if args.threshold_db > 0:
        errors.append(f"Práh musí být záporný nebo nulový: {args.threshold_db}")
    if args.min_length <= 0:
        errors.append(f"Minimální délka musí být kladná: {args.min_length}")
    if args.min_length_after_trim <= 0:
        errors.append(f"Min délka po trimu musí být kladná: {args.min_length_after_trim}")
    if errors:
        for e in errors:
            logger.error(e)
        return False
    return True

# --- Audio načítání ---
def get_audio_info(file_path):
    try:
        with wave.open(file_path, 'rb') as wf:
            frames = wf.getnframes()
            sr = wf.getframerate()
            ch = wf.getnchannels()
            sw = wf.getsampwidth()
            duration = frames / sr
            return {'sample_rate': sr, 'channels': ch, 'sample_width': sw,
                    'frames': frames, 'duration': duration,
                    'bit_depth': sw*8, 'type': 'mono' if ch==1 else 'stereo'}
    except Exception as e:
        logger.error(f"Nelze načíst info {file_path}: {e}")
        return None

def load_audio_data(file_path, info):
    try:
        with wave.open(file_path, 'rb') as wf:
            raw_data = wf.readframes(info['frames'])
        if info['bit_depth']==16:
            dtype, scale = np.int16, 32767.0
        elif info['bit_depth']==32:
            dtype, scale = np.int32, 2147483647.0
        else:
            logger.error(f"Nepodporovaná bit depth {info['bit_depth']}")
            return None
        data = np.frombuffer(raw_data, dtype=dtype)
        if info['channels'] > 1:
            data = data.reshape(-1, info['channels'])
        data_float = data.astype(np.float32)/scale
        return data_float, scale, dtype
    except Exception as e:
        logger.error(f"Chyba při načítání {file_path}: {e}")
        return None

# --- Uložení segmentu ---
def save_audio_segment(data, output_path, info, scale, dtype):
    try:
        if dtype == np.int16:
            data_int = (data*scale).clip(-32768,32767).astype(np.int16)
        elif dtype==np.int32:
            data_int = (data*scale).clip(-2147483648,2147483647).astype(np.int32)
        else:
            data_int = data.astype(dtype)
        with wave.open(output_path,'wb') as wf:
            wf.setnchannels(info['channels'])
            wf.setsampwidth(info['sample_width'])
            wf.setframerate(info['sample_rate'])
            wf.writeframes(data_int.tobytes())
        return True
    except Exception as e:
        logger.error(f"Chyba při ukládání {output_path}: {e}")
        return False

# --- Segmentace podle energie ---
def detect_segments(data, fs, threshold_db=-40, min_segment_length=3):
    if len(data)==0: return []
    max_abs = np.max(np.abs(data))
    if max_abs==0: return []
    hop = int(0.05*fs)
    num_windows = len(data)//hop
    if num_windows==0: return []
    reshaped = data[:num_windows*hop].reshape(-1,hop)
    rms = np.sqrt(np.mean(reshaped**2,axis=1))
    rms_db = 20*np.log10(rms/max_abs + 1e-10)
    active = rms_db > threshold_db
    segments, start_idx = [], None
    for i, a in enumerate(active):
        if a and start_idx is None:
            start_idx=i
        elif not a and start_idx is not None:
            end_idx=i
            if (end_idx-start_idx)*hop/fs >= min_segment_length:
                segments.append((start_idx*hop, end_idx*hop))
            start_idx=None
    if start_idx is not None:
        end_idx = len(data)
        if (end_idx - (start_idx*hop))/fs >= min_segment_length:
            segments.append((start_idx*hop, end_idx))
    return segments

# --- Trim ticha ---
def trim_silence(data, fs, threshold_db, window_size=0.01):
    if len(data)==0: return np.array([]),0,0
    mono = np.mean(data,axis=1) if data.ndim>1 else data
    max_amp = np.max(np.abs(mono))
    if max_amp==0:
        return np.array([]),0,0
    hop = max(1,int(window_size*fs))
    num_win = len(mono)//hop
    if num_win==0: return np.array([]),0,0
    reshaped = mono[:num_win*hop].reshape(-1,hop)
    rms = np.sqrt(np.mean(reshaped**2,axis=1))
    rms_db = 20*np.log10(rms/max_amp + 1e-10)
    active = np.where(rms_db>=threshold_db)[0]
    if len(active)==0: return np.array([]),0,0
    start, end = active[0]*hop, min((active[-1]+1)*hop,len(data))
    return data[start:end], start, end

# --- Fine-trim začátku podle náběhu energie ---
def fine_trim_start(data, fs, fine_threshold=0.25, window_size=0.01, min_active_windows=2):
    """
    Trim začátku podle bloků zvýšené energie.
    - window_size: délka okna pro RMS v sekundách
    - fine_threshold: relativní práh proti max RMS v segmentu
    - min_active_windows: počet po sobě jdoucích oken nad prahem, které definují začátek
    """
    if len(data) == 0:
        return data, 0
    mono = np.mean(data, axis=1) if data.ndim > 1 else data
    win_samples = max(1, int(window_size * fs))

    # RMS po oknech
    rms = np.array([np.sqrt(np.mean(mono[i:i + win_samples] ** 2))
                    for i in range(0, len(mono) - win_samples + 1, win_samples)])

    # Prah jako frakce max RMS
    threshold = fine_threshold * np.max(rms)

    # Hledání první sekvence oken nad prahem
    active = np.where(rms >= threshold)[0]
    if len(active) == 0:
        return data, 0

    # Najdeme první blok alespoň min_active_windows dlouhý
    for i in range(len(active) - min_active_windows + 1):
        if active[i + min_active_windows - 1] - active[i] == min_active_windows - 1:
            trim_start = active[i] * win_samples
            return data[trim_start:], trim_start

    # Pokud žádný blok, použijeme první aktivní okno
    trim_start = active[0] * win_samples
    return data[trim_start:], trim_start


# --- Fade-out podle amplitudy ---
def auto_fadeout(segment_data, fs, drop_ratio=0.7):
    """Fade-out lineárně až na konec segmentu podle poklesu amplitudy."""
    if len(segment_data)==0: return segment_data
    mono = np.mean(segment_data,axis=1) if segment_data.ndim>1 else segment_data
    max_amp = np.max(np.abs(mono))
    threshold = max_amp * drop_ratio
    start_idx = None
    for i in range(len(mono)):
        if np.abs(mono[i]) < threshold:
            start_idx = i
            break
    if start_idx is None or start_idx>=len(segment_data):
        return segment_data
    fade_len = len(segment_data)-start_idx
    fade_curve = np.linspace(1.0,0.0,fade_len)
    if segment_data.ndim==1:
        segment_data[start_idx:]*=fade_curve
    else:
        segment_data[start_idx:,:]*=fade_curve[:,np.newaxis]
    return segment_data

# --- Zpracování WAV souboru ---
def process_wav(input_path, output_dir, threshold_db, min_length, min_length_after_trim,
                trim_threshold_offset, apply_fades, stats, fadeout_ratio=0.7):
    base_name = Path(input_path).stem
    logger.info(f"Načítám soubor: {Path(input_path).name}")
    info = get_audio_info(input_path)
    if not info:
        stats.files_failed += 1
        return
    stats.add_format(info['sample_rate'], info['channels'], info['bit_depth'])
    stats.total_input_duration += info['duration']
    result = load_audio_data(input_path, info)
    if not result:
        stats.files_failed += 1
        return
    data, scale, dtype = result
    data -= np.mean(data,axis=0) if data.ndim>1 else np.mean(data)

    mono = np.mean(data,axis=1) if data.ndim>1 else data
    segments = detect_segments(mono, info['sample_rate'], threshold_db, min_length)
    logger.info(f"Nalezeno {len(segments)} segmentů.")
    if not segments:
        stats.files_processed +=1
        return

    trim_db = threshold_db + trim_threshold_offset
    segments_saved=0

    for idx,(s_start,s_end) in enumerate(tqdm(segments,desc=f"Segmenty {base_name}",leave=False)):
        seg = data[s_start:s_end].copy()
        trimmed, t_start, t_end = trim_silence(seg, info['sample_rate'], trim_db)
        if len(trimmed)==0:
            stats.segments_skipped +=1
            continue
        trimmed,fine_s = fine_trim_start(trimmed, info['sample_rate'])
        t_start += fine_s
        if len(trimmed)/info['sample_rate'] < min_length_after_trim:
            stats.segments_skipped +=1
            continue
        if apply_fades:
            trimmed = auto_fadeout(trimmed, info['sample_rate'], drop_ratio=fadeout_ratio)
        start_ms = int((s_start+t_start)/info['sample_rate']*1000)
        dur_ms = int(len(trimmed)/info['sample_rate']*1000)
        fmt = f"{info['sample_rate']//1000}k_{info['type']}"
        fname = f"{base_name}_slice_{idx+1:03d}_start_{start_ms}ms_dur_{dur_ms}ms_{fmt}.wav"
        out_path = Path(output_dir)/fname
        counter=1
        while out_path.exists():
            fname = f"{base_name}_slice_{idx+1:03d}_start_{start_ms}ms_dur_{dur_ms}ms_{fmt}_{counter}.wav"
            out_path = Path(output_dir)/fname
            counter+=1
        if save_audio_segment(trimmed,str(out_path),info,scale,dtype):
            segments_saved+=1
            stats.segments_created+=1
            stats.total_output_duration+=len(trimmed)/info['sample_rate']
        else:
            stats.segments_skipped+=1

    logger.info(f"Uloženo {segments_saved}/{len(segments)} segmentů z {base_name}")
    stats.files_processed+=1

# --- Main ---
def main():
    parser = argparse.ArgumentParser(description="Rozdělení WAV souborů na segmenty s fine-trimem a fade-out podle amplitudy")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold_db", type=float, default=-45)
    parser.add_argument("--min_length", type=float, default=3.0)
    parser.add_argument("--min_length_after_trim", type=float, default=0.5)
    parser.add_argument("--trim_threshold_offset", type=float, default=10)
    parser.add_argument("--fadeout_threshold_ratio", type=float, default=0.7,
                        help="Pokles amplitudy pro spuštění fade-out (0-1)")
    parser.add_argument("--no_fades", action="store_true")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()

    logger.setLevel(getattr(logging,args.log_level.upper()))
    if not validate_inputs(args):
        return 1
    Path(args.output_dir).mkdir(parents=True,exist_ok=True)
    wav_files = list(Path(args.input_dir).glob("*.[wW][aA][vV]"))
    if not wav_files:
        logger.error(f"Žádné WAV soubory v {args.input_dir}")
        return 1
    logger.info(f"Nalezeno {len(wav_files)} WAV souborů.")

    if args.preview:
        logger.info("=== PREVIEW ===")
        for wf in wav_files:
            info = get_audio_info(str(wf))
            if info:
                logger.info(f"{wf.name}: {info['sample_rate']}Hz, {info['channels']}ch, {info['bit_depth']}bit, {info['duration']:.2f}s")
        return 0

    stats = ProcessingStats()
    apply_fades = not args.no_fades

    for wf in tqdm(wav_files,desc="Zpracovávám soubory",unit="soubor"):
        if args.resume:
            base = wf.stem
            existing = list(Path(args.output_dir).glob(f"{base}_slice_*.wav"))
            if existing:
                logger.info(f"Přeskakuji {wf.name} - existují výstupy")
                continue
        process_wav(str(wf), args.output_dir, args.threshold_db,
                    args.min_length, args.min_length_after_trim,
                    args.trim_threshold_offset, apply_fades, stats,
                    fadeout_ratio=args.fadeout_threshold_ratio)

    stats.print_summary()
    logger.info("Zpracování dokončeno.")
    return 0

if __name__ == "__main__":
    exit(main())
