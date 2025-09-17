# Audio Slicer

Program pro automatické rozdělení WAV souborů na jednotlivé audio segmenty na základě detekce hlasitosti. Nástroj identifikuje části zvuku nad definovaným prahem hlasitosti a ukládá je jako samostatné WAV soubory s časovými timestampy. **Vylepšená verze** podporuje různé vzorkovací frekvence, bit depths a automatickou detekci mono/stereo formátů.

## Funkce

- **Automatická detekce segmentů**: Identifikuje aktivní audio segmenty nad nastavitelným prahem hlasitosti
- **Inteligentní ořezávání**: Odstraňuje ticho na začátku a konci každého segmentu s konfigurovatelným prahem
- **Podpora různých formátů**: 
  - Vzorkovací frekvence: 44.1kHz, 48kHz a další
  - Bit depth: 16-bit, 32-bit (částečně 24-bit)
  - Automatická detekce mono/stereo
- **Zachování originálních parametrů**: Výstupní soubory mají stejné audio parametry jako vstup
- **Fade-in/fade-out**: Konfigurovatelné fade (výchozí 5ms) pro odstranění kliknutí
- **Časové timestampy**: Výstupní soubory obsahují informaci o původní pozici v ms a délce
- **Pokročilé funkce**:
  - Preview mód pro testování nastavení
  - Resume mód pro pokračování přerušeného zpracování
  - Detailní statistiky zpracování
- **Optimalizace výkonu**: Vektorizované výpočty pro rychlé zpracování velkých souborů
- **Progress tracking**: Ukazatel průběhu zpracování s tqdm

## Použití

### Základní příkaz
```bash
python slicer.py --input-dir vstupni_adresar --output-dir vystupni_adresar
```

### Kompletní příklad s parametry
```bash
python slicer.py --input-dir samples_in --output-dir samples_out_sliced --threshold_db -45 --min_length 3 --fade_ms 5
```

### Preview mód (testování bez zpracování)
```bash
python slicer.py --input-dir samples_in --output-dir samples_out --preview
```

## Parametry

### Povinné parametry
- `--input-dir`: Cesta k adresáři obsahujícímu vstupní WAV soubory
- `--output-dir`: Cesta k adresáři pro uložení rozdělených segmentů

### Volitelné parametry
- `--threshold_db`: Práh detekce hlasitosti v dB (výchozí: -45 dB)
- `--min_length`: Minimální délka segmentu v sekundách (výchozí: 3.0)
- `--min_length_after_trim`: Minimální délka segmentu po ořezání ticha v sekundách (výchozí: 0.5)
- `--trim_threshold_offset`: Offset pro práh ořezávání vzhledem k detekčnímu prahu v dB (výchozí: +10)
- `--fade_ms`: Délka fade-in/fade-out v milisekundách (výchozí: 5.0)
- `--no_fades`: Zakáže fade-in/fade-out (může způsobit kliknutí)
- `--resume`: Přeskočí již existující výstupní soubory
- `--preview`: Preview mód - zobrazí co by se stalo bez skutečného zpracování
- `--log_level`: Úroveň logování - DEBUG, INFO, WARNING, ERROR (výchozí: INFO)

## Požadavky

```
numpy
tqdm
```

## Instalace

1. Naklonujte repository:
```bash
git clone <repository-url>
cd audio-slicer
```

2. Nainstalujte závislosti:
```bash
pip install numpy tqdm
```

## Formát vstupních souborů

Program podporuje WAV soubory s následujícími specifikacemi:

- **Bit depth**: 16-bit, 32-bit integer (částečně 24-bit s fallback)
- **Kanály**: Mono nebo stereo (automatická detekce)
- **Vzorkovací frekvence**: Libovolná (44.1 kHz, 48 kHz, 96 kHz, atd.)
- **Délka**: Bez omezení

## Formát výstupních souborů

Rozdělené segmenty jsou uloženy s názvem ve formátu:

```
{original_name}_slice_{number}_start_{timestamp}ms_dur_{duration}ms_{format}.wav
```

### Příklady názvů souborů:
- `piano01_slice_001_start_1250ms_dur_3500ms_48k_stereo.wav` - první segment začínající v čase 1.25s, trvající 3.5s, 48kHz stereo
- `recording_slice_003_start_45680ms_dur_2100ms_44k_mono.wav` - třetí segment začínající v čase 45.68s, trvající 2.1s, 44.1kHz mono
- `audio_slice_002_start_8900ms_dur_1800ms_48k_stereo_1.wav` - druhý segment s číslovaným sufixem při kolizi názvů

### Formát informace v názvu:
- `48k` = 48kHz vzorkovací frekvence
- `44k` = 44.1kHz vzorkovací frekvence  
- `mono` / `stereo` = počet kanálů

## Algoritmus zpracování

### 1. Validace a analýza vstupů
- **Detekce formátu**: Automatická identifikace vzorkovací frekvence, počtu kanálů a bit depth
- **Validace parametrů**: Kontrola platnosti vstupních hodnot
- **Statistiky**: Sledování formátů a celkového zpracování

### 2. Detekce segmentů
- **RMS analýza**: Výpočet efektivní hodnoty (RMS) v klouzavých oknech adaptivních podle vzorkovací frekvence
- **Práh detekce**: Identifikace aktivních částí nad definovaným prahem v dB
- **Spojování**: Spojení blízkých aktivních oblastí do souvislých segmentů

### 3. Preprocessing
- **DC offset removal**: Odstranění stejnosměrné složky pro každý kanál samostatně
- **Mono konverze pro analýzu**: Převod stereo na mono pouze pro detekci (zachování stereo ve výstupu)
- **Float32 konverze**: Přesné výpočty bez přetečení s návratem do původního formátu

### 4. Ořezávání segmentů
- **Konfigurovatelný práh**: Práh pro ořezávání je nastavitelný offset nad detekčním prahem (výchozí +10 dB)
- **Adaptivní okno**: Velikost okna pro ořezávání se přizpůsobuje vzorkovací frekvenci
- **Minimální délka**: Filtrování příliš krátkých segmentů po ořezání (výchozí < 0.5s)

### 5. Post-processing
- **Fade-in/fade-out**: Aplikace krátkých fade přechodů (výchozí 5ms) pro odstranění kliknutí
- **Zachování kvality**: Výstup ve stejném formátu jako vstup (bit depth, vzorkovací frekvence, kanály)

### 6. Výstup
- **Informativní názvy**: Obsahují časové informace a formát
- **Unikátní názvy**: Automatické číslování při kolizi názvů souborů
- **Statistiky**: Detailní report o zpracování

## Příklady použití

### 1. Základní rozdělení s výchozími parametry
```bash
python slicer.py --input-dir ./samples_in --output-dir ./samples_out_sliced
```

### 2. Citlivější detekce pro tišší nahrávky s kratším fade
```bash
python slicer.py --input-dir ./samples_in --output-dir ./samples_out_sliced --threshold_db -50 --min_length 2 --fade_ms 3
```

### 3. Zpracování se speciálními parametry ořezávání
```bash
python slicer.py --input-dir ./samples_in --output-dir ./samples_out_sliced --trim_threshold_offset 15 --min_length_after_trim 1.0
```

### 4. Preview mód s debug informacemi
```bash
python slicer.py --input-dir ./samples_in --output-dir ./samples_out_sliced --preview --log_level DEBUG
```

### 5. Resume zpracování bez fade efektů
```bash
python slicer.py --input-dir ./samples_in --output-dir ./samples_out_sliced --resume --no_fades
```

## Výstup programu

Program zobrazuje průběh zpracování s podrobnými informacemi a statistikami:

```
INFO: Nalezeno 3 WAV souborů k zpracování.
Zpracovávám soubory: 100%|████████████| 3/3 [00:15<00:00,  5.12s/soubor]
INFO: Načítám soubor: piano_recording.wav
INFO: Formát: 48000Hz, 2 kanál(y), 16-bit
INFO: Délka: 120.50s
INFO: Detekuji segmenty...
INFO: Nalezeno 8 segmentů.
Segmenty z piano_recording.wav: 100%|████████████| 8/8 [00:03<00:00,  2.5segment/s]
INFO: Uložen segment 1: piano_recording_slice_001_start_1250ms_dur_3500ms_48k_stereo.wav
INFO: Uložen segment 2: piano_recording_slice_002_start_8900ms_dur_2100ms_48k_stereo.wav
INFO: Segment 3 je po ořezání příliš krátký (0.385s), přeskakuji.
INFO: Uloženo 7/8 segmentů z piano_recording.wav

=== SOUHRN ZPRACOVÁNÍ ===
INFO: Soubory: 3 úspěšné, 0 neúspěšné
INFO: Segmenty: 19 vytvořené, 4 přeskočené
INFO: Doba: 280.5s vstup → 95.2s výstup
INFO: Formáty:
INFO:   48000Hz_2ch_16bit: 2 souborů
INFO:   44100Hz_1ch_16bit: 1 souborů
INFO: Zpracování dokončeno.
```

## Řešení problémů

### Běžné chyby

**"Chyba při načítání souboru"**
- Ověřte, že soubor je validní WAV formát
- Zkontrolujte, zda není soubor poškozen
- Program nyní podporuje různé bit depths - zkuste různé soubory

**"Segment je po ořezání prázdný"**
- Snižte `threshold_db` hodnotu pro citlivější detekci
- Zkontrolujte `trim_threshold_offset` - možná je příliš vysoký
- Použijte `--preview` pro analýzu před zpracováním

**"Nalezeno 0 segmentů"**
- Audio je příliš tiché nebo práh je příliš vysoký
- Zkuste `--threshold_db -60` nebo nižší
- Ověřte formát souboru v preview módu

**"Nepodporovaná bit depth"**
- Program podporuje 16-bit a 32-bit
- 24-bit soubory se automaticky převedou na 16-bit
- Použijte audio editor pro konverzi nekompatibilních formátů

### Tipy pro optimální výsledky

1. **Testování parametrů**: Použijte `--preview` pro otestování nastavení na malém vzorku
2. **Kvalitní vstup**: Používejte čisté nahrávky s minimálním šumem
3. **Správný práh**: Experimentujte s `threshold_db` pro vaše konkrétní nahrávky
4. **Ořezávání**: Upravte `trim_threshold_offset` podle potřeby zachování/odstranění tichých částí
5. **Fade délka**: Pro perkusní nástroje možná snižte `--fade_ms` na 2-3ms
6. **Resume funkce**: Pro velké dávky použijte `--resume` při přerušení
7. **Monitoring**: Používejte `DEBUG` log level pro diagnostiku problémů

## Technické detaily

- **Multi-format support**: Automatická detekce a zachování audio parametrů
- **Adaptive processing**: Algoritmy se přizpůsobují vzorkovací frekvenci
- **Vektorizované výpočty**: NumPy operace pro vysoký výkon
- **Memory efficient**: Postupné zpracování bez načítání všech dat do paměti

