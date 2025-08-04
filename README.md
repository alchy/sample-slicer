# Audio Slicer

Program pro automatické rozdělení WAV souborů na jednotlivé audio segmenty na základě detekce hlasitosti. Nástroj identifikuje části zvuku nad definovaným prahem hlasitosti a ukládá je jako samostatné WAV soubory s časovými timestampy.

## Funkce

- **Automatická detekce segmentů**: Identifikuje aktivní audio segmenty nad nastavitelným prahem hlasitosti
- **Inteligentní ořezávání**: Odstraňuje ticho na začátku a konci každého segmentu
- **Podpora stereo i mono**: Zpracovává jak stereo, tak mono WAV soubory
- **Časové timestampy**: Výstupní soubory obsahují informaci o původní pozici v ms
- **Konfigurovatelné parametry**: Nastavitelný práh detekce a minimální délka segmentů
- **Optimalizace výkonu**: Vektorizované výpočty pro rychlé zpracování velkých souborů
- **Progress tracking**: Ukazatel průběhu zpracování s tqdm

## Použití

### Základní příkaz
```bash
python slicer.py --input-dir vstupni_adresar --output-dir vystupni_adresar
```

### Kompletní příklad s parametry
```bash
python slicer.py --input-dir samples_in --output-dir samples_out_sliced --threshold_db -45 --min_length 3
```

## Parametry

### Povinné parametry
- `--input-dir`: Cesta k adresáři obsahujícímu vstupní WAV soubory
- `--output-dir`: Cesta k adresáři pro uložení rozdělených segmentů

### Volitelné parametry
- `--threshold_db`: Práh detekce hlasitosti v dB (výchozí: -45 dB)
- `--min_length`: Minimální délka segmentu v sekundách (výchozí: 3 sekundy)
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
- **Formát**: 16-bit WAV soubory
- **Kanály**: Mono nebo stereo
- **Vzorkovací frekvence**: Libovolná (nejčastěji 44.1 kHz, 48 kHz)
- **Délka**: Bez omezení

## Formát výstupních souborů

Rozdělené segmenty jsou uloženy s názvem ve formátu:
```
{original_name}_slice_{number}_start_{timestamp}ms.wav
```

### Příklady názvů souborů:
- `song01_slice_1_start_1250ms.wav` - první segment začínající v čase 1.25 sekundy
- `recording_slice_3_start_45680ms.wav` - třetí segment začínající v čase 45.68 sekundy
- `audio_slice_2_start_8900ms_1.wav` - druhý segment s číslovaným sufixem při kolizi názvů

## Algoritmus zpracování

### 1. Detekce segmentů
- **RMS analýza**: Výpočet efektivní hodnoty (RMS) v klouzavých oknech
- **Práh detekce**: Identifikace aktivních částí nad definovaným prahem v dB
- **Spojování**: Spojení blízkých aktivních oblastí do souvislých segmentů

### 2. Preprocessing
- **DC offset removal**: Odstranění stejnosměrné složky
- **Mono konverze**: Převod stereo na mono pro analýzu (zachování stereo ve výstupu)
- **Float32 konverze**: Přesné výpočty bez přetečení

### 3. Ořezávání segmentů
- **Adaptivní práh**: Práh pro ořezávání je +10 dB nad detekčním prahem
- **Jemné okno**: 10ms okna pro přesné ořezání ticha
- **Minimální délka**: Filtrování příliš krátkých segmentů (< 0.1s po ořezání)

### 4. Výstup
- **Zachování kvality**: 16-bit výstup se zachováním původních audio parametrů
- **Unikátní názvy**: Automatické číslování při kolizi názvů souborů

## Příklady použití

### 1. Základní rozdělení s výchozími parametry
```bash
python slicer.py --input-dir ./samples_in --output-dir ./samples_out_sliced
```

### 2. Citlivější detekce pro tišší nahrávky
```bash
python slicer.py --input-dir ./samples_in --output-dir ./samples_out_sliced --threshold_db -40 --min_length 5
```

### 3. Debug režim s podrobným logováním
```bash
python slicer.py --input-dir ./samples_in --output-dir ./samples_out_sliced --log_level DEBUG
```

## Výstup programu

Program zobrazuje průběh zpracování s podrobnými informacemi:

```
INFO: Nalezeno 3 WAV souborů k zpracování.
Zpracovávám soubory: 100%|████████████| 3/3 [00:15<00:00,  5.12s/soubor]
INFO: Načítám soubor: recording001.wav
INFO: Očišťuji DC offset...
INFO: Detekuji segmenty...
INFO: Nalezeno 5 segmentů.
Zpracovávám segmenty z recording001.wav: 100%|████████████| 5/5 [00:02<00:00,  2.1segment/s]
INFO: Uložen segment 1: recording001_slice_1_start_1250ms.wav
INFO: Uložen segment 2: recording001_slice_2_start_8900ms.wav
INFO: Segment 3 je po ořezání příliš krátký (0.085s), přeskakuji.
INFO: Uložen segment 4: recording001_slice_4_start_25600ms.wav
INFO: Uložen segment 5: recording001_slice_5_start_42300ms.wav
INFO: Zpracování dokončeno.
```

## Řešení problémů

### Běžné chyby

**"Podporovány pouze 16-bit WAV soubory"**
- Převeďte soubory na 16-bit formát pomocí audio editoru
- Použijte nástroje jako FFmpeg: `ffmpeg -i input.wav -acodec pcm_s16le output.wav`

**"Segment je po ořezání prázdný"**
- Snižte threshold_db hodnotu pro citlivější detekci
- Zkontrolujte, zda soubor obsahuje dostatečně hlasitý zvuk

**"Nalezeno 0 segmentů"**
- Audio je příliš tiché nebo práh je příliš vysoký
- Zkuste threshold_db -60 nebo nižší
- Ověřte, že WAV soubor obsahuje audio data

### Tipy pro optimální výsledky

1. **Kvalitní vstup**: Používejte čisté nahrávky s minimálním šumem
2. **Správný práh**: Experimentujte s threshold_db pro vaše konkrétní nahrávky
3. **Testování**: Vyzkoušejte parametry na malém vzorku před velkým batchem
4. **Monitoring**: Používejte DEBUG log level pro diagnostiku problémů

## Technické detaily

- **Vektorizované výpočty**: NumPy operace pro vysoký výkon
- **Memory efficient**: Postupné zpracování bez načítání všech dat do paměti
- **Thread-safe logging**: Kompatibilní s tqdm progress bary
- **Robustní error handling**: Pokračování zpracování při chybách jednotlivých souborů

## Limitace

- Podporuje pouze 16-bit WAV soubory
- Optimalizováno pro jednoduchý mono/stereo obsah
- Není vhodné pro komplexní polyfónní audio s překrývajícími se segmenty

