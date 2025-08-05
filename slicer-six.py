"""
python slicer6.py --input-dir samples_in --output-dir samples_out_sliced4 --threshold-db -40  --multi-onset --log-level DEBUG --onset-rel-threshold 0.5
"""

import os
import numpy as np
import wave
import argparse
import logging
import sys
from tqdm import tqdm


class LoggerSetup:
    @staticmethod
    def init_logging(level=logging.INFO):
        class TqdmStreamHandler(logging.StreamHandler):
            def emit(self, record):
                try:
                    msg = self.format(record)
                    tqdm.write(msg, file=sys.stderr)
                    self.flush()
                except Exception:
                    self.handleError(record)

        logger = logging.getLogger(__name__)
        logger.setLevel(level)

        if logger.hasHandlers():
            logger.handlers.clear()

        handler = TqdmStreamHandler()
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger


class PianoSlicer:
    def __init__(self, input_dir, output_dir, threshold_db=-45.0, release_time=2.5,
                 max_length=12.0, multi_onset=False, onset_sensitivity=0.9,
                 min_gap=0.15, onset_rel_threshold=0.5, logger=None):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.threshold_db = threshold_db
        self.release_time = release_time
        self.max_length = max_length
        self.multi_onset = multi_onset
        self.onset_sensitivity = onset_sensitivity
        self.min_gap = min_gap
        self.onset_rel_threshold = onset_rel_threshold
        self.logger = logger or logging.getLogger(__name__)

    def process(self):
        os.makedirs(self.output_dir, exist_ok=True)
        wav_files = [f for f in os.listdir(self.input_dir) if f.lower().endswith(".wav")]

        self.logger.info(f"Nalezeno {len(wav_files)} WAV souborů ke zpracování.")

        for filename in tqdm(wav_files, desc="Zpracovávám soubory", unit="soubor"):
            input_path = os.path.join(self.input_dir, filename)
            self._process_single_file(input_path)

    def _process_single_file(self, input_path):
        base_name = os.path.splitext(os.path.basename(input_path))[0]
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
            self.logger.error(f"Chyba při načítání souboru {os.path.basename(input_path)}: {e}")
            return

        if data.ndim == 1:
            data = data[:, np.newaxis]

        data = data.astype(np.float32)
        data -= np.mean(data, axis=0)
        mono_data = np.mean(data, axis=1)

        segments = self._detect_piano_segments(mono_data, fs)
        if self.multi_onset:
            segments = self._apply_multi_onset(mono_data, segments, fs)

        self.logger.info(f"Nalezeno {len(segments)} segmentů.")

        for idx, (start, end) in enumerate(segments):
            mono_segment = mono_data[start:end]
            stereo_segment = data[start:end, :]

            trimmed_mono, trimmed_stereo = self._smart_trim(mono_segment, stereo_segment, fs)
            if len(trimmed_mono) / fs < 0.1:
                self.logger.debug(f"Segment {idx + 1} je příliš krátký po ořezu, přeskakuji.")
                continue

            start_ms = int(start / fs * 1000)
            duration_ms = int(len(trimmed_mono) / fs * 1000)
            filename = f"{base_name}_slice_{idx + 1:03d}_start_{start_ms}ms_dur_{duration_ms}ms.wav"
            output_path = os.path.join(self.output_dir, filename)

            try:
                with wave.open(output_path, 'wb') as wf:
                    wf.setnchannels(trimmed_stereo.shape[1])
                    wf.setsampwidth(2)
                    wf.setframerate(fs)
                    wf.writeframes(trimmed_stereo.astype(np.int16).tobytes())
                self.logger.info(f"Uložen segment: {filename}")
            except Exception as e:
                self.logger.error(f"Chyba při ukládání segmentu {filename}: {e}")

    def _detect_piano_segments(self, mono_data, fs):
        window_size = 0.05
        hop = int(fs * window_size)
        threshold = self.threshold_db
        release_time = int(self.release_time / window_size)
        max_windows = int(self.max_length / window_size)
        release_threshold = threshold - 15

        rms = self._rms_windows(mono_data, hop)
        rms_db = 20 * np.log10(rms / (np.max(np.abs(mono_data)) + 1e-10))

        segments = []
        onset = None
        last_active = None

        for i, level in enumerate(rms_db):
            if level > threshold and onset is None:
                onset = i
                last_active = i
            elif onset is not None:
                if level > release_threshold:
                    last_active = i
                if i - onset >= max_windows:
                    segments.append((onset * hop, min((onset + max_windows) * hop, len(mono_data))))
                    onset = None
                elif i - last_active >= release_time:
                    end = min((last_active + 1) * hop, len(mono_data))
                    if (end - onset * hop) / fs >= 0.1:
                        segments.append((onset * hop, end))
                    onset = None

        if onset is not None:
            end = min((last_active + 1) * hop, len(mono_data))
            if (end - onset * hop) / fs >= 0.1:
                segments.append((onset * hop, end))

        return segments

    def _apply_multi_onset(self, mono_data, segments, fs):
        final_segments = []
        for start, end in segments:
            sub_segments = self._detect_multiple_onsets(mono_data[start:end], fs)
            for sub_start, sub_end in sub_segments:
                abs_start = start + sub_start
                abs_end = start + sub_end
                if (abs_end - abs_start) / fs >= 0.1:
                    final_segments.append((abs_start, abs_end))
        return final_segments

    def _detect_multiple_onsets(self, segment, fs):
        window_size = 0.02
        hop = int(fs * window_size)
        rms = self._rms_windows(segment, hop)
        rms /= (np.max(rms) + 1e-10)

        threshold = 10 ** ((self.threshold_db + (1 - self.onset_sensitivity) * 10) / 20)
        min_gap = int(self.min_gap * fs / hop)

        onsets = [0]
        current_max = rms[0]

        for i in range(1, len(rms)):
            increase = (rms[i] - current_max) / (current_max + 1e-10)
            if rms[i] > threshold and increase > self.onset_rel_threshold and i - onsets[-1] >= min_gap:
                onsets.append(i)
                current_max = rms[i]
            elif rms[i] > current_max * 0.9:
                current_max = max(current_max, rms[i])

        sub_segments = []
        for i in range(len(onsets)):
            start = onsets[i] * hop
            end = (onsets[i + 1] * hop if i + 1 < len(onsets) else len(segment))
            sub_segments.append((start, end))

        return sub_segments

    def _rms_windows(self, data, hop):
        trim = len(data) - len(data) % hop
        return np.sqrt(np.mean(data[:trim].reshape(-1, hop) ** 2, axis=1))

    def _smart_trim(self, mono_segment, stereo_segment, fs):
        trim_window = int(fs * 0.01)
        rms = self._rms_windows(mono_segment, trim_window)
        rms_db = 20 * np.log10(rms / (np.max(np.abs(stereo_segment)) + 1e-10))
        active = np.where(rms_db >= self.threshold_db + 5)[0]
        if len(active) == 0:
            return mono_segment, stereo_segment
        first = max(0, (active[0] - 1) * trim_window)
        return mono_segment[first:], stereo_segment[first:]


def parse_arguments():
    parser = argparse.ArgumentParser(description="Detekce a řezání piano samplů z WAV souborů.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--threshold-db", type=float, default=-45.0)
    parser.add_argument("--release-time", type=float, default=2.5)
    parser.add_argument("--max-length", type=float, default=12.0)
    parser.add_argument("--multi-onset", action="store_true")
    parser.add_argument("--onset-sensitivity", type=float, default=0.9)
    parser.add_argument("--min-gap", type=float, default=0.15)
    parser.add_argument("--onset-rel-threshold", type=float, default=0.5)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main():
    args = parse_arguments()
    logger = LoggerSetup.init_logging(getattr(logging, args.log_level.upper()))
    slicer = PianoSlicer(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        threshold_db=args.threshold_db,
        release_time=args.release_time,
        max_length=args.max_length,
        multi_onset=args.multi_onset,
        onset_sensitivity=args.onset_sensitivity,
        min_gap=args.min_gap,
        onset_rel_threshold=args.onset_rel_threshold,
        logger=logger
    )
    slicer.process()


if __name__ == "__main__":
    main()
