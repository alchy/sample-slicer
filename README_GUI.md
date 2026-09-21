# sample-slicer GUI

Qt okno nad stejným balíčkem `sample_slicer`, které používá CLI. Dělá totéž co
`sample-slicer analyze / build / slice`, jen s uloženými profily a logem v okně.

## Spuštění

```bash
.venv/bin/pip install -e '.[gui]'      # PySide6 + platformdirs
sample-slicer-gui                      # nebo: python -m slicergui
```

## Okno shora dolů

1. **Profil** – pojmenovaná sada složek a parametrů (typicky jeden profil na
   banku). *Uložit*, *Uložit jako…*, *Smazat*. Profily jsou v jednom JSON
   souboru v uživatelské konfiguraci (`platformdirs`, na macOS
   `~/Library/Application Support/sample-slicer/profiles.json`); při startu se
   načte naposledy uložený.
2. **Režim** – *Banka pro ithaca* (analyze / build) nebo *Generický střih*
   (slice). Podle režimu se ukážou jen relevantní složky a tlačítka.
3. **Složky** – Zdrojové nahrávky; pro banku Original (96 kHz / 24 bit +
   report + index) a Banka pro ithaca (48 kHz / 16 bit); pro střih Výstup.
   Ke každé je nápověda v poli.
4. **Parametry** – jen ty, které se v praxi ladí: Konec dozvuku (dBFS),
   Umělý dozvuk (s), Max délka samplu (s), Pre-roll (ms), Fade-in (ms) a
   přepínač *Doladit na temperované ladění*. Ostatní prahy detekce mají
   výchozí hodnoty (viz `docs/algorithm.md`); kdo je potřebuje, použije CLI (`--help`).
5. **Akce** – *Analyzovat (dry-run)* vypíše tabulku úderů (čas, délka, peak,
   nota, centy, confidence, verdikt) do logu a nic nezapisuje; *Sestavit banku*
   spustí celý workflow; *Otevřít report* otevře `report.md` z Original.
   Ve střihu je místo toho *Rozřezat*.
6. **Log** – průběh úlohy; úloha běží v samostatném vlákně, okno zůstává
   responzivní.

Ruční zásahy (`overrides.json` ve zdrojové složce) a idempotence (index
v Original) fungují stejně jako v CLI – viz `README.md`.
