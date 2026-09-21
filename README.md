# sample-slicer

Nástroj pro rozřezání nahrávek nástroje (typicky piana) na jednotlivé samply a
pro stavbu banky pro sample player **ithaca-legacy**. Ze zdrojového adresáře
surových WAV nahrávek (např. 96 kHz / 24 bit ze stereo páru mikrofonů) udělá:

- ořezané údery s korektním nasazením a přirozeným dozvukem do nuly,
- u každého úderu určí **notu jen z audia** (bez znalosti pořadí nahrávání),
- zapíše banku ve formátu `m###/<hash>.wav` v originálu i v 48 kHz / 16 bit,
- vede index (opakovaný běh nic nezdvojí) a report pro kontrolu člověkem.

Jak to uvnitř funguje: `docs/algorithm.md`. Jak nahrávat, aby banka dopadla dobře: `RECORDING.md`.

## Funkce

- **Detekce úderů** na RMS obálce s lokálním šumovým dnem (nahrávky klavíru
  nemají ploché ticho mezi tóny – basová struna zní 40 s a víc).
- **Nasazení = attack struny**, ne mechanika klávesy, která zní 40–60 ms před
  ním (jinak by sample nesl latenci a odhad výšky seděl v šumu). Pre-roll 5 ms,
  fade-in 2 ms.
- **Dělení slitých tónů** a **ořez pádu kladívka/dusítka** po uvolnění klávesy;
  obojí jen na transientu, aby zázněje basových sborů neplatily za nový úder.
- **Konec samplu** se *vyrábí*: surová data končí na `end_level` (-60 dBFS,
  resp. lokální dno + 6 dB) nebo před artefaktem, pak exponenciální dozvuk
  navazující na naměřený sklon poklesu až do nuly. Dozvuk nikdy nesahá za
  nasazení dalšího tónu.
- **Odhad výšky**: autokorelace s normalizací pásma („zrychlení" signálu přes
  k = ¼ … 16, bez převzorkování), hlasování přes k, spektrální důkaz parciál,
  doladění na fundamentálu z plného sample rate. Na reálném Petrofu 53/53 not
  správně od A0 po C8, bez oktávových chyb.
- **Ladicí křivka**: piano není temperované přesně (Petrof: A7–C8 +46 až +79 c);
  noty se přiřazují postupně vůči křivce vyhlazené z jistých kotev.
- **WAV 16/24/32 bit**, mono i stereo, libovolný sample rate; výstup zachovává
  formát vstupu. Float WAV se odmítne s hláškou.
- Idempotentní `build` s indexem, `_rejected/` pro nejisté údery,
  `overrides.json` pro ruční zásahy, `report.md`.

## Instalace

Python ≥ 3.11, numpy, tqdm; pro `build` navíc **ffmpeg** v PATH.

```bash
git clone https://github.com/alchy/sample-slicer.git
cd sample-slicer
python3.11 -m venv .venv
.venv/bin/pip install -e .          # + '.[gui]' pro Qt GUI, '.[dev]' pro pytest
```

## Použití

### Stavba banky pro ithaca-legacy

```bash
sample-slicer analyze <raw-dir> [--truth truth.json]      # dry-run, nic nezapisuje
sample-slicer build <raw-dir> --original <orig> --out <bank>
```

- `<orig>/m###/<hash>.wav` – ořezané údery v původním formátu (např. 96 kHz / 24 bit)
- `<bank>/m###/<hash>.wav` – 48 kHz / 16 bit (ffmpeg soxr, bez libsoxr swresample; + dither), tohle načítá ithaca
- `<orig>/report.md` – tabulka všech úderů (čas, délka, peak, nota, centy, confidence), odmítnuté, vrstvy na notu, díry na klaviatuře, ladění
- `<orig>/_rejected/` – údery bez spolehlivé noty (nízká confidence, mimo toleranci, mimo klavír)
- `<orig>/.slicer-index.json` – idempotence: zdroj se stejným obsahem a parametry se přeskočí, nové nahrávky se přidají, změna parametrů nahradí staré soubory; cizí soubory v bance se nikdy nemažou
- `<raw-dir>/overrides.json` – ruční zásahy: `{"rec.wav": {"skip": [17], "midi": {"3": 24}}}`
- `--retune` – posune výšku každého úderu na temperované ladění (změnou poměru resamplingu); výchozí vypnuto

Nota se určuje jen z audia. `--truth` slouží výhradně k měření přesnosti proti
známému pořadí nahrávání (`analyze` vypíše ok / oktávová chyba / jiná po oktávách):

```json
{"260917_0180.wav": {"start": "A0", "pattern": "chromatic"},
 "260917_0179.wav": {"start": "C4", "pattern": "major"},
 "test.wav":        {"pattern": "list", "notes": ["C4", "E4", "G4"]}}
```

Opakované údery téže noty jsou další velocity vrstvy (ithaca je seřadí podle
naměřeného RMS), pipeline nic nenormalizuje.

### Generický střih (bez not)

```bash
sample-slicer slice <vstupni_adresar> <vystupni_adresar>
```

Výstup: `{zdroj}_slice_{NNN}_start_{ms}ms_dur_{ms}ms.wav` ve formátu vstupu.
`python slicer.py slice A B` je totéž bez instalace konzolového příkazu. Staré
přepínače (`--input-dir`, `--threshold_db`, `--min_length`, …) už neexistují.

### Parametry detekce

Všechny podpříkazy sdílejí přepínače (`--help` vypíše výchozí hodnoty):

| přepínač | výchozí | význam |
|---|---|---|
| `--end-level-db` | -60 | úroveň (dBFS), pod kterou surová data končí; efektivně max(hodnota, lokální dno + 6 dB) |
| `--tail-s` | 2 | délka umělého dozvuku do nuly |
| `--max-len-s` | 30 | horní limit délky samplu |
| `--onset-rise-db` / `--onset-rise-ms` | 20 / 30 | skok nad lokální dno, který znamená úder |
| `--onset-peak-within-db` | 10 | nasazení = první rámec do X dB od vrcholu attacku |
| `--preroll-ms` / `--fade-in-ms` | 5 / 2 | kolik vzít před nasazením, fade-in |
| `--split-rise-db`, `--split-peak-within-db`, `--split-min-len-s` | 12, 20, 0.5 | dělení slitých tónů |
| `--artifact-rise-db`, `--artifact-after-s`, `--artifact-window-s` | 8, 1, 2 | pád kladívka: skok nad regresní čáru dozvuku |
| `--click-below-peak-db` | 25 | úder slabší o X dB než nejhlasitější v souboru = klik |

Příklad: kratší basové samply (dozvuk končí na -50 dB):

```bash
sample-slicer build raw/ --original orig/ --out bank/ --end-level-db -50
```

## Formát vstupních souborů

WAV PCM int 16 / 24 / 32 bit, mono nebo stereo, libovolný sample rate. Analýza
běží na mono mixu 0,5·(L+R) po odečtení DC offsetu; do výstupu jde původní
stereo. Float WAV není podporován (chyba pro daný soubor, pokračuje se dalším).

## Výstup programu

```
$ sample-slicer build raw/ --original orig/ --out bank/
UPOZORNĚNÍ: ffmpeg bez libsoxr — resampling přes swresample (filter_size=256)
260917_0179.wav: 29 zapsáno, 0 odmítnuto
260917_0180.wav: 24 zapsáno, 0 odmítnuto
Zapsáno 53 úderů, odmítnuto 0, přeskočeno zdrojů 0. Report: orig/report.md
```

Druhý běh stejného příkazu: `Zapsáno 0 úderů, … přeskočeno zdrojů 2`.

## Řešení problémů

- **Úder skončil v `_rejected/`** – v `report.md` je důvod: `low_confidence`
  (šum, dva tóny naráz, příliš krátký úsek), `out_of_tolerance` (výška mimo
  ±50 c od ladicí křivky), `out_of_range` (mimo A0–C8). Když víš, co to je,
  přidej do `overrides.json` buď `skip`, nebo správné `midi`.
- **Dva tóny v jednom samplu** – druhý úder byl tišší o víc než 20 dB, nebo
  přišel dřív než 200 ms po prvním. Zkus `--split-peak-within-db 30`.
- **Sample začíná pozdě / cvakne** – nasazení řídí `--onset-peak-within-db`
  (větší hodnota = dřívější start) a `--preroll-ms`.
- **Basové samply jsou dlouhé (20–30 s)** – to je skutečný dozvuk struny k
  -60 dB. Kratší: `--end-level-db -50`.
- **Šumové kliky se počítají jako údery** – jsou o víc než 25 dB pod
  nejhlasitějším úderem? Pak je vyřadí `--click-below-peak-db`; jinak
  `overrides.json`.
- **`ffmpeg nenalezen`** – `build` ho potřebuje pro převod na 48 kHz / 16 bit
  (`brew install ffmpeg`); `analyze` a `slice` bez něj fungují.

## Vývoj

```bash
.venv/bin/pip install -e '.[dev]'
.venv/bin/pytest -q
```

Testy obsahují regresní laťku na výřezech z reálných nahrávek Petrof
(`tests/fixtures/pitch`, 53 úderů A0–G#2 a C4–C8 s pravdou v `truth.json`),
syntetické testy detekce (nasazení ±5 ms, slité tóny, thump, kliky), dozvuku,
I/O 24 bit a idempotence stavby banky. Fixtures regeneruje
`tools/make_fixtures.py <raw-dir> <truth.json> tests/fixtures/pitch`.

Struktura balíčku `sample_slicer/`: `io` (WAV), `envelope` (obálka, dno,
sklon), `detect` (segmenty), `tail` (dozvuk), `slicing` (generický střih),
`pitch` (výška), `notes` (názvy not, pravda), `tuning` (ladicí křivka,
přiřazení), `analyze` (dry-run), `bank` (stavba banky), `cli`.

## GUI

`sample-slicer-gui` (po `pip install -e '.[gui]'`) je Qt okno nad týmž balíčkem:
profil na banku, režim banka / střih, složky, pět praktických parametrů,
tlačítka Analyzovat / Sestavit banku / Otevřít report a log. Viz `README_GUI.md`.
