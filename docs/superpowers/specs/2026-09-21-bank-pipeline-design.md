# Návrh: pipeline nahrávka → banka samplů pro ithaca-legacy

Datum: 2026-09-21 · Stav: schválený návrh, čeká na implementační plán

## 1. Cíl a hranice

Jeden příkaz, který ze zdrojového adresáře surových nahrávek piana postaví
dynamic-velocity banku pro sample player ithaca-legacy:

- vstup: adresář WAV souborů (96 kHz / 24 bit / stereo, TASCAM DR-40X + pár
  AUDIX SCX25); každý soubor je řada úderů jednotlivých kláves, jedna velocity
  na soubor, mezi údery několik desítek sekund; soubory přibývají postupně,
- výstup 1: `<original>/m###/<hash>.wav` – ořezané údery v původním formátu,
- výstup 2: `<out>/m###/<hash>.wav` – totéž převedené na 48 kHz / 16 bit,
- vedlejší výstupy: index (idempotence), report (kontrola člověkem),
  `_rejected/` (údery, které se nepodařilo přiřadit).

Nástroj **nezná pořadí nahrávání**. Notu určuje jen z audia. Známé pořadí
existujících nahrávek slouží pouze jako testovací pravda (`--truth`).

Co je mimo rozsah: packed banka (`soundbank.ithaca`, řeší `bake_soundbank.py`
v ithace), multi-mic, round-robin, více velocity v jedné nahrávce.

**ithaca-legacy se nemění.** Pouze načítá výsledný adresář; formát
dynamic-velocity banky (složka `m###/`, libovolný název WAV, vrstvy řazené
podle naměřeného peak RMS při načtení) je daný a tímto návrhem se neupravuje.

## 2. Umístění a závislosti

Celý workflow žije v repu **sample-slicer** (github.com/alchy/sample-slicer),
které zůstává samostatné. Refaktor stávajícího `slicer.py` na balíček
`sample_slicer` (`pyproject.toml`, konzolový příkaz `sample-slicer`).

Závislosti: Python ≥ 3.11, numpy, tqdm. Externí: **ffmpeg** s resamplerem
soxr (ověří se při startu; chybí-li, jasná chyba). GUI (`slicergui`, PySide6)
zůstává jako volitelná vrstva nad tímž API.

Struktura:

```
sample_slicer/
  io.py        čtení/zápis WAV 16/24/32 bit (PCM int), DC offset per kanál, stereo zachováno
  envelope.py  RMS obálka, lokální šumové dno, sklon dozvuku
  detect.py    nasazení, dělení slitých tónů, konec, artefakt uvolnění → list[Segment]
  tail.py      přirozený dozvuk do nuly, fade-in
  pitch.py     odhad výšky → Pitch(f0, midi, cents, confidence)
  tuning.py    ladicí křivka, přiřazení noty
  bank.py      layout, index, ffmpeg převod, report
  cli.py       podpříkazy analyze / build / slice
slicer.py, slicergui/   tenká vrstva nad detect/tail (původní generické chování)
tests/                  fixtures + pytest
```

Každý modul má jeden účel a čisté vstupy/výstupy (numpy pole + parametry),
aby šel testovat bez souborů.

## 3. Detekce úderu (`envelope.py`, `detect.py`)

Analýza probíhá na mono mixu (0,5·(L+R)) po odečtení DC offsetu; do výstupu
jde původní stereo.

**Obálka.** RMS v rámcích 5 ms (nasazení) a 10 ms (zbytek), v dB.

**Lokální šumové dno.** Nahrávky nemají ploché ticho mezi tóny – basová struna
zní 40 s a víc. Dno se proto neměří globálně, ale jako 5. percentil obálky
v okně 2 s těsně před nasazením daného úderu. Všechny prahy jsou relativní
k němu, pokud není řečeno jinak.

**Nasazení.** Úder začíná tam, kde obálka během 30 ms vyskočí o víc než
20 dB nad lokální dno. Od tohoto bodu se jde zpět k lokálnímu minimu obálky;
začátek segmentu = toto minimum minus 5 ms pre-roll. Víc pre-rollu ne
(latence při hraní). Fade-in 2 ms.

**Slité tóny.** Uvnitř aktivního úseku je další skok o víc než 12 dB během
30 ms kandidát na nový úder. Rozdělí se, jen když vrchol nové části je do
20 dB od vrcholu předchozího úderu a nová část trvá aspoň 0,5 s (odliší druhý
tón od pádu dusítka).

**Falešné údery.** Segment s vrcholem víc než 25 dB pod nejhlasitějším úderem
souboru se zahodí (šumové kliky), zapíše se do reportu jako odmítnutý.

**Konec segmentu** = první z událostí:

1. obálka klesne pod `end_level` (výchozí -60 dBFS, absolutní, parametr),
2. artefakt uvolnění klávesy: po první sekundě dozvuku se lineární regresí
   přes poslední 2 s sleduje sklon poklesu (dB/s); skok o víc než 8 dB nad
   tuto čáru = pád kladívka/dusítka → konec v lokálním minimu těsně před ním,
3. horní limit délky (výchozí 30 s),
4. nasazení dalšího úderu.

Surová data za bodem konce se nikdy nepoužijí (hráč mohl klávesu pustit,
zatímco struna neslyšitelně zněla).

**Přirozený dozvuk do nuly (`tail.py`).** Od bodu konce se aplikuje
exponenciální útlum navazující na naměřený sklon: strmost = max(naměřený
sklon, minimální sklon tak, aby se za `tail_s` (výchozí 2 s) dosáhlo -96 dB
vůči úrovni v bodě konce); posledních 10 ms lineárně do nuly. Výsledný sample
je bod konce + `tail_s`.

## 4. Odhad výšky (`pitch.py`)

Ověřeno spikem na obou existujících nahrávkách (0180: A0–G#2 chromaticky,
24 úderů; 0179: C dur C4–C8, 29 úderů). Výsledek spiku: 24/24 a 27/29
správně přiřazeno s pevnou tolerancí ±50 c; zbylé dva údery (B7, C8) jsou
změřené správně, ale piano je tam +79/+73 c – to řeší ladicí křivka (§5),
laťka 29/29 se ověří testem.

Analyzuje se úsek od nasazení + 40 ms.

1. **Normalizace pásma („zrychlení").** Pro k ∈ {¼, ½, 1, 2, 4, 8, 16} se
   signál interpretuje s nominálním sample rate `sr·k` (bez převzorkování) a
   spočítá se normalizovaná autokorelace okna 250 ms nominálně (tj. 250·k ms
   reálně), po horní propusti 20 Hz. Hledá se vrchol v nominálním pásmu
   100–1600 Hz; musí být vnitřní lokální maximum (hraniční vrcholy se
   zahazují), poloha se zpřesní parabolou. Odhad = f/k, váha = výška vrcholu;
   vrcholy pod 0,6 se ignorují.
2. **Hlasování.** Odhady se shlukují po ±50 centech; skóre shluku = součet
   čtverců vah.
3. **Spektrální důkaz.** Spektrum 0,5 s od nasazení (Hann, zero-padding 8×).
   Pro každý shluk se spočítá prominence parciál 1–4 (max v ±60 c kolem h·f0
   minus medián log-spektra v ±1 oktávě). Shluk s méně než dvěma prominentními
   parciálami (≥ 12 dB) dostane skóre ×0,1. Tím vypadne úder kladívka
   (~90 Hz), který u krátkých diskantových tónů vítězí v autokorelaci.
   Fundamentál se nevyžaduje (u A0 je v spektru ~1 dB nad okolím).
4. **Doladění.** Konečné f0 se čte spektrálně z plného sample rate: nejnižší
   prominentní parciála h, vrchol v ±150 c kolem h·f0, parabolická interpolace
   v log-magnitudě, děleno h. Vyšší parciály se nepoužijí, když je fundamentál
   vidět (nehармonicita v diskantu posouvá 2. parciálu o +40 až +80 c).
5. **Confidence** = skóre vítěze / součet skóre všech shluků, spolu s počtem
   souhlasících k a prominencí; hlásí se v reportu.

Výstup: `Pitch(f0_hz, midi_float, confidence, n_votes, evidence)`.

Co se nepoužije (změřeno, selhává): prostá HPS, YIN, autokorelace bez
normalizace pásma, harmonická sumace jako jediný odhad (oktávové chyby A0–C#1).

## 5. Přiřazení noty (`tuning.py`)

Piano není temperované přesně (Petrof: bas -30 až +9 c, střed ±20 c, A7–C8
+46 až +79 c). Pevná tolerance ±50 c by horní klávesy zahodila.

1. **První průchod:** údery, jejichž `midi_float` je do ±35 c od celé noty a
   confidence je vysoká, se přiřadí rovnou (jisté kotvy).
2. **Ladicí křivka:** z kotev se vyhladí odchylka (centy) jako funkce MIDI
   (klouzavý medián přes ±6 půltónů, lineární interpolace mezi kotvami,
   konstantní extrapolace na krajích).
3. **Druhý průchod:** ostatní údery se přiřadí k nejbližší notě po odečtení
   křivky; přijme se odchylka do ±50 c od křivky. Mimo → `_rejected/`.
4. Rozsah klavíru 21–108; mimo → odmítnuto.
5. Kolize (dvě přiřazení téže noty z jednoho souboru) se nezakazuje: jsou to
   opakované údery, tedy další vrstvy (ithaca je seřadí podle RMS).

Report vypíše odchylku každého úderu od temperovaného ladění i od křivky.

## 6. Zápis banky (`bank.py`)

- `<original>/m###/<hash>.wav`: 96 kHz / 24 bit / stereo, žádná normalizace
  hlasitosti (naměřené RMS řídí vrstvy v ithace). Hash = prvních 16 hex znaků
  MD5 obsahu souboru (stejná konvence jako `make_dynamic_bank.sh`).
- `<out>/m###/<hash>.wav`: ffmpeg
  `-af aresample=48000:resampler=soxr:precision=28:dither_method=triangular
  -c:a pcm_s16le`; hash z výsledného souboru.
- Volitelně `--retune`: před převodem se výška posune o naměřenou odchylku od
  temperovaného ladění změnou poměru resamplingu (vypnuto výchozí; piano se
  před finálním samplováním ladí).
- **Index** `<original>/.slicer-index.json`: pro každý (zdroj, pořadí úderu):
  hash original, hash out, MIDI, centy, confidence, čas ve zdroji, verze
  nástroje a hash parametrů. Opakovaný běh: zdroj se stejným obsahem a
  parametry se přeskočí; změněné parametry → staré soubory téhož úderu se
  smažou a nahradí. Ruční soubory v bance, které index nezná, se nikdy nemažou.
- **Report** `<original>/report.md`: tabulka úderů (zdroj, čas, délka, peak
  dBFS, MIDI, centy vs. temperované, centy vs. křivka, confidence, verdikt);
  oddíl odmítnutých s důvodem; souhrn: počet vrstev na notu, díry na
  klaviatuře, tabulka ladění.
- `<original>/_rejected/<zdroj>_<index>_<důvod>.wav`.
- **Ruční zásah** `overrides.json` ve zdrojovém adresáři:
  `{"260917_0180.wav": {"skip": [17], "midi": {"3": 24}}}`. Override má
  přednost před detekcí a v reportu je označen.

## 7. CLI (`cli.py`)

- `sample-slicer analyze <src-dir> [--truth truth.json] [parametry]` –
  dry-run: vypíše segmenty, noty, centy, confidence; s `--truth` přesnost po
  oktávách (OK / oktávová chyba / jiná). Nic nezapisuje.
- `sample-slicer build <src-dir> --original <dir> --out <dir> [--retune]
  [parametry]` – celý workflow.
- `sample-slicer slice <in-dir> <out-dir> [parametry]` – původní generické
  chování (segmenty bez not, názvy s časem), pro jiné než pianové použití.

Formát `truth.json`: `{"260917_0180.wav": {"start": "A0", "pattern":
"chromatic"}, "260917_0179.wav": {"start": "C4", "pattern": "major"}}`;
`pattern` ∈ {chromatic, major, list} (u `list` výčet not).

Všechny prahy z §3 a §4 jsou parametry s uvedenými výchozími hodnotami.

## 8. Chyby a hlášení

- Nepodporovaný WAV (float, ≠ PCM int 16/24/32) → chyba pro daný soubor,
  pokračuje se dalším; v reportu.
- Chybějící ffmpeg → chyba při startu `build`, `analyze` funguje.
- Nízká confidence nebo odchylka mimo toleranci → `_rejected/`, nikdy tiché
  přiřazení.
- Nic se nemaže mimo soubory, které index sám vytvořil.

## 9. Testy (`tests/`, pytest)

- **Pitch regrese:** fixtures = výřezy 0,6 s od nasazení každého úderu z 0180
  a 0179 v původním formátu, jen mono (24 + 29 souborů, ~170 kB každý,
  ~9 MB celkem), pravda v JSON. Laťka: 24/24 a 29/29 s ladicí křivkou,
  žádná oktávová chyba.
- **Detekce na syntetice:** signál se známými nasazeními, dvěma slitými tóny,
  umělým „thumpem" v dozvuku a kliky pod prahem; kontrola začátků (±5 ms),
  rozdělení, konce před thumpem, zahození kliků.
- **Tail:** obálka výstupu monotónně klesá, končí nulou, sklon ≥ naměřený.
- **I/O:** round-trip 24 bit bit-přesně, stereo, DC offset.
- **Idempotence:** `build` dvakrát → identický adresář a index; změna
  parametru → staré soubory nahrazeny, cizí soubor zachován.
- **Tuning:** křivka z kotev + přiřazení s +79 c v diskantu.

## 10. Postup implementace (pořadí etap)

1. `io` + `envelope` + `detect` + `tail` a přepojení `slicer.py` (generický
   střih s korektním 24 bit, nasazením a dozvukem).
2. `pitch` + `tuning` + `analyze --truth`, regresní fixtures, laťka 24/24 a
   29/29.
3. `bank` + `build` (original, ffmpeg převod, index, report, rejected,
   overrides) + testy idempotence.
4. Načtení `ap-petrof-dynamic` v ithaca-gui jako ruční kontrola; README
   sample-sliceru popíše workflow.

## 11. Poznámky pro další nahrávání (mimo software)

- Vrcholy v 0180 jsou -10 až -24 dBFS; odstup od šumu u tichých tónů ~40 dB.
  Nahrávat o 6–10 dB hlasitěji.
- Bas zní přes 40 s; sample při -60 dBFS bude 20–25 s. Klávesu buď držet do
  neslyšitelnosti, nebo pustit vědomě dřív (pravidlo artefaktu ji odřízne).
- Vrchní oktáva je +46 až +79 c; piano se před finálním samplováním naladí.
