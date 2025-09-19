# WAV Segmenter s Fine-Trimem a Fade-Out

Tento nástroj umožňuje rozdělit WAV soubory na samostatné segmenty podle energie signálu, automaticky odstraňuje ticho na začátku segmentu (fine-trim) a umožňuje aplikovat fade-out na konci segmentu.

---

## Funkce

- **Detekce segmentů podle energie:**  
  Segmenty se vytvářejí na základě RMS energie signálu a zvoleného prahu (`threshold_db`).  
- **Fine-trim začátku segmentu:**  
  Trimuje ticho na začátku segmentu a nastavuje start na první blok stabilně zvýšené energie.  
- **Trim ticha:**  
  Odstraňuje tiché části segmentu podle RMS energie.  
- **Fade-out:**  
  Lineární fade-out od bodu, kde amplituda klesne pod definovanou hodnotu (`fadeout_threshold_ratio`).  
- **Podpora vícestopého audio:**  
  Stereo i mono soubory jsou správně zpracovány.

---

## Instalace

```bash
git clone <repo-url>
cd <repo>
pip install -r requirements.txt
```

Vyžaduje Python 3.8+ a knihovny numpy a tqdm.

```
python segmenter.py --input-dir ./input --output-dir ./output \
    --threshold_db -45 --min_length 3.0 --min_length_after_trim 0.5 \
    --trim_threshold_offset 10 --fadeout_threshold_ratio 0.7
```

| Parametr                    | Popis                                             | Výchozí hodnota |
| --------------------------- | ------------------------------------------------- | --------------- |
| `--threshold_db`            | Prah RMS energie pro detekci segmentů (dB)        | -45             |
| `--min_length`              | Minimální délka segmentu (s)                      | 3.0             |
| `--min_length_after_trim`   | Minimální délka segmentu po trimu (s)             | 0.5             |
| `--trim_threshold_offset`   | Offset prahu pro trim ticha (dB)                  | 10              |
| `--fadeout_threshold_ratio` | Pokles amplitudy pro spuštění fade-out (0–1)      | 0.7             |
| `--no_fades`                | Nepoužít fade-out                                 | False           |
| `--log_level`               | Úroveň logování                                   | INFO            |
| `--resume`                  | Pokračovat v zpracování existujících souborů      | False           |
| `--preview`                 | Zobrazit informace o WAV souborech bez zpracování | False           |

| Parametr                | Popis                                           | Doporučená hodnota |
| ----------------------- | ----------------------------------------------- | ------------------ |
| `fine_threshold`        | Frakce maximální RMS pro detekci aktivních oken | 0.25               |
| `window_size`           | Délka okna RMS v sekundách                      | 0.02 s             |
| `min_active_windows`    | Počet po sobě jdoucích oken nad prahem          | 4                  |
| `post_trigger_shift_ms` | Posun trimu do segmentu (ms)                    | 5–10               |




