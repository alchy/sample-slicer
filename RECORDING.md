# Nahrávací protokol pro banku

Co se ukázalo na prvních dvou řadách Petrofu (0180: A0–G#2, 0179: C dur
C4–C8) a co zlevní další nahrávání víc než jakákoli úprava softwaru.

## Úroveň

- Vrcholy úderů byly -10 až -24 dBFS, odstup od šumu u tichých tónů jen
  ~40 dB. **Nahrávej o 6–10 dB hlasitěji** (vrcholy kolem -6 dBFS u nejsilnější
  dynamiky), bez limiteru a bez automatické úrovně na rekordéru – klouzavý
  gain by rozbil řazení velocity vrstev podle RMS.
- Jedna nahrávka = jedna dynamika. Nástroj štítek „mf/ff" nepotřebuje, vrstvy
  vzniknou z naměřeného RMS; ale v jedné řadě má být úhoz co nejstejnější.

## Řada

- Pořadí kláves je libovolné (nota se určuje z audia), ale souvislá
  chromatická řada se nejlíp kontroluje: `analyze --truth` pak spočítá
  přesnost. Pro pokrytí celé klaviatury je potřeba i černé klávesy – C dur
  z 0179 nechala 22 děr.
- Mezi údery nech aspoň 3 s ticha nad úrovní dozvuku; bas zní přes 40 s, tam
  je rozestup 35–45 s v pořádku (sample se stejně utne na -60 dBFS,
  tj. 12–30 s).
- Opakovaný úder téže klávesy je v pořádku – uloží se jako další vrstva.

## Klávesa a dozvuk

- Buď drž klávesu, dokud tón nedozní pod slyšitelnost, nebo ji pusť vědomě
  dřív – pád kladívka/dusítka je transient a pravidlo artefaktu ho odřízne.
  Nepouštěj ji „někde uprostřed“ potichu s prodlevou: struna pak zní dál a
  konec samplu je jen vyrobený dozvuk.
- Mechanika klávesy zní 40–60 ms před strunou; nástroj ji vynechává (sample
  začíná attackem struny). Netřeba se snažit hrát „tiše mechanicky“.

## Ladění

- Vrchní oktáva Petrofu byla +46 až +79 c nad temperovaným laděním. Nástroj
  to zvládne (ladicí křivka), ale ithaca za běhu nepřelaďuje, takže samply
  hrají tak, jak je piano naladěné. **Před finální řadou piano naladit.**
  Nouzově existuje `--retune` (posun výšky změnou poměru resamplingu).

## Kontrola po nahrání

1. `sample-slicer analyze <raw> --truth truth.json` – počet úderů sedí,
   0 oktávových chyb, confidence > 0,6.
2. `sample-slicer build …` a poslech v ithaca-gui: nasazení bez cvaknutí,
   konec bez pádu kladívka, žádný cizí tón uvnitř samplu.
3. `report.md`: díry na klaviatuře, vrstvy na notu, ladění.
