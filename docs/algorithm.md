# Jak sample-slicer funguje

Živý popis implementace (as-built). Vznikl z návrhu z 2026-09-21 a průběžně se
upravuje podle kódu; když se mění `detect`, `pitch`, `tuning` nebo `bank`, mění
se i příslušná sekce tady. Nahrávací doporučení jsou v `RECORDING.md`.

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
podle naměřeného peak RMS při načtení) je daný a tento nástroj se mu přizpůsobuje.

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

**Obálka.** RMS v rámcích 5 ms, v dB; pro rozhodnutí o konci navíc vyhlazená
klouzavým mediánem 100 ms (`env_s`). Artefakty a nasazení se čtou ze surové
obálky, protože medián by krátký transient smazal.

**Lokální šumové dno.** Nahrávky nemají ploché ticho mezi tóny – basová struna
zní 40 s a víc. Dno se proto neměří globálně, ale jako 5. percentil obálky
v okně 2 s těsně před nasazením daného úderu. Všechny prahy jsou relativní
k němu, pokud není řečeno jinak.

**Nasazení.** Kandidát je tam, kde surová obálka během 30 ms vyskočí o víc než
20 dB nad lokální dno. Skutečné nasazení je pak **první rámec, který se dostane
do 10 dB od vrcholu následujících 200 ms** – tedy attack struny. Mechanika
klávesy zní 40–60 ms před strunou o 20–30 dB slaběji; brát ji za začátek by
přidalo latenci při hraní a posunulo analýzu výšky do šumu. Začátek segmentu =
nasazení minus 5 ms pre-roll. Fade-in 2 ms.

**Slité tóny.** Uvnitř aktivního úseku je další skok o víc než 12 dB během
30 ms kandidát na nový úder. Rozdělí se, jen když (a) od posledního nasazení
uplynulo víc než 200 ms (attack sám o sobě není nový úder), (b) nový vrchol
převýší nedávné maximum obálky o 3 dB (zázněje basových sborů se houpou pod
ním), (c) je do 20 dB od vrcholu předchozího úderu a (d) nová část drží aspoň
0,5 s (odliší druhý tón od pádu dusítka).

**Falešné údery.** Segment s vrcholem víc než 25 dB pod nejhlasitějším úderem
souboru se zahodí (šumové kliky), zapíše se do reportu jako odmítnutý.

**Konec segmentu** = první z událostí (kontrola začíná hned za vrcholem
attacku, ne po pevné době):

1. vyhlazená obálka klesne pod `max(end_level, lokální dno + 6 dB)`
   (`end_level` výchozí -60 dBFS, parametr) – lokální dno chrání před tím, aby
   sample vlekl šum místnosti, když je v té části nahrávky nad -60 dB,
2. artefakt uvolnění klávesy: po první sekundě dozvuku se lineární regresí
   přes poslední 2 s vyhlazené obálky sleduje sklon poklesu; **rychlý** skok
   (> 8 dB během 30 ms v surové obálce) o víc než 8 dB nad regresní čáru = pád
   kladívka/dusítka → konec v lokálním minimu 100 ms před ním. Pomalé +8 dB
   houpání záznějů (stovky ms) pravidlo nespustí,
3. horní limit délky (výchozí 30 s),
4. nasazení dalšího úderu (segment končí před jeho pre-rollem).

Každý segment nese i **tvrdý limit** (index nasazení dalšího úderu): render za
něj nikdy nesáhne, ani umělým dozvukem.

Surová data za bodem konce se nikdy nepoužijí (hráč mohl klávesu pustit,
zatímco struna neslyšitelně zněla).

**Přirozený dozvuk do nuly (`tail.py`).** Od bodu konce se aplikuje
exponenciální útlum navazující na naměřený sklon: strmost = max(naměřený
sklon, minimální sklon tak, aby se za `tail_s` (výchozí 2 s) dosáhlo -96 dB
vůči úrovni v bodě konce); posledních 10 ms lineárně do nuly. Výsledný sample
je bod konce + `tail_s`.

## 4. Odhad výšky (`pitch.py`)

Ověřeno spikem na obou existujících nahrávkách (0180: A0–G#2 chromaticky,
24 úderů; 0179: C dur C4–C8, 29 úderů). Výsledek implementace (`analyze
--truth`): 53/53 správně, 0 oktávových chyb; B7 a C8 jsou +79/+73 c (piano),
což řeší ladicí křivka (§5).

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
   minus medián log-spektra v ±1 oktávě). Shluk musí mít aspoň dvě prominentní
   parciály (≥ 12 dB) **a z nich aspoň jednu lichou** (1. nebo 3.; subharmonický
   kandidát f0/2 má reálné jen sudé), jinak skóre ×0,1. Navíc spojitá váha
   min(1, Σ clip(prominence, 0, 40) / 100): úder kladívka a rezonance místnosti
   (~110 Hz, u krátkých diskantových tónů vítězí v autokorelaci) nemají silnou
   harmonickou řadu. Fundamentál se nevyžaduje (u A0 je v spektru ~1 dB nad
   okolím).
4. **Doladění.** Konečné f0 se čte spektrálně z plného sample rate: nejnižší
   prominentní parciála h, vrchol v ±150 c kolem h·f0, parabolická interpolace
   v log-magnitudě, děleno h. Vyšší parciály se nepoužijí, když je fundamentál
   vidět (nehармonicita v diskantu posouvá 2. parciálu o +40 až +80 c).
5. **Confidence** = skóre vítěze / (skóre vítěze + Σ skóre shluků, které nejsou
   jeho celočíselným násobkem či podílem 2–4). Sub/superharmonické shluky jsou u
   klavíru vždy přítomné a rozhoduje o nich autokorelace; do nejistoty se počítá
   jen nesouvisející konkurence (šum, rezonance, druhý tón). Na reálných datech
   nejnižší 0,62; práh přijetí 0,5.

Analyzuje se jen úsek nasazení → konec segmentu (bez umělého dozvuku: v něm u
krátkých tónů přežívá jen rezonance).

Výstup: `Pitch(f0_hz, midi_float, confidence, n_votes, evidence)`.

Co se nepoužije (změřeno, selhává): prostá HPS, YIN, autokorelace bez
normalizace pásma, harmonická sumace jako jediný odhad (oktávové chyby A0–C#1).

## 5. Přiřazení noty (`tuning.py`)

Piano není temperované přesně (Petrof: bas -30 až +9 c, střed ±20 c, A7–C8
+46 až +79 c). Pevná tolerance ±50 c by horní klávesy zahodila.

1. **Kotvy:** údery, jejichž `midi_float` je do ±35 c od celé noty a confidence je
   vysoká. Každá kotva se ověří proti křivce z *ostatních* kotev (leave-one-out):
   odchylka > 35 c = nejspíš sousední nota s velkou odchylkou → není kotva.
2. **Ladicí křivka:** z kotev se vyhladí odchylka (centy) jako funkce MIDI
   (klouzavý medián přes ±2 půltóny, lineární interpolace mezi kotvami,
   **lineární extrapolace** z krajních dvou bodů se sklonem omezeným na
   25 c/půltón – stretch v krajních oktávách roste strmě, konstanta by B7/C8
   nechytila).
3. **Postupné přiřazení:** ze zbývajících úderů se vždy vezme ten nejblíže
   celé notě po odečtení křivky; přijme se do ±50 c od křivky, přidá se mezi
   kotvy a křivka se přepočítá. Odmítnuté (mimo toleranci) → `_rejected/`.
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
  -c:a pcm_s16le`; hash z výsledného souboru. Když ffmpeg nemá libsoxr
  (Homebrew build), použije se swresample s `filter_size=256:cutoff=0.98`
  (ověří se jednou při startu, hlásí se varováním, engine je zapsán v indexu).
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

- **Pitch regrese:** fixtures = výřezy od nasazení každého úderu z 0180 a
  0179 v původním formátu, jen mono (1,5 s pod C3 kvůli hlasům k = 8, jinak
  0,6 s; 24 + 29 souborů, ~15 MB celkem), pravda v JSON. Laťka: správná
  oktáva a do ±90 c u všech 53 (`tests/test_pitch_fixtures.py`); přiřazení
  včetně ladicí křivky se ověřuje na reálných hodnotách v `test_tuning.py`
  a end-to-end přes `analyze --truth` (53/53).
- **GUI:** profily bez Qt (`test_profiles.py`), offscreen smoke test okna
  (`test_gui_smoke.py`, přeskočí se bez PySide6).
- **Detekce na syntetice:** signál se známými nasazeními, dvěma slitými tóny,
  umělým „thumpem" v dozvuku a kliky pod prahem; kontrola začátků (±5 ms),
  rozdělení, konce před thumpem, zahození kliků.
- **Tail:** obálka výstupu monotónně klesá, končí nulou, sklon ≥ naměřený.
- **I/O:** round-trip 24 bit bit-přesně, stereo, DC offset.
- **Idempotence:** `build` dvakrát → identický adresář a index; změna
  parametru → staré soubory nahrazeny, cizí soubor zachován.
- **Tuning:** křivka z kotev + přiřazení s +79 c v diskantu.
