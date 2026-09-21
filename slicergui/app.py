"""Hlavní okno: profil → režim → složky → parametry → akce → log."""
from __future__ import annotations
import subprocess, sys
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QLabel,
                               QLineEdit, QPushButton, QComboBox, QRadioButton, QDoubleSpinBox, QCheckBox,
                               QPlainTextEdit, QFileDialog, QMessageBox, QInputDialog)
from .profiles import ProfileStore, DEFAULT_PARAMS
from .worker import JobWorker


class MainWindow(QMainWindow):
    def __init__(self, store: ProfileStore | None = None):
        super().__init__()
        self.store = store or ProfileStore()
        self.worker: JobWorker | None = None
        self.setWindowTitle("sample-slicer")
        self.setMinimumSize(860, 640)
        self._build_ui()
        self._load_profile(self.store.last)

    # ---- UI --------------------------------------------------------------
    def _build_ui(self):
        root = QWidget(); self.setCentralWidget(root)
        lay = QVBoxLayout(root)

        # profil
        prof = QHBoxLayout()
        prof.addWidget(QLabel("Profil:"))
        self.profile_box = QComboBox(); self.profile_box.setMinimumWidth(220)
        self.profile_box.addItems(self.store.names())
        self.profile_box.currentTextChanged.connect(self._load_profile)
        prof.addWidget(self.profile_box)
        for text, slot in (("Uložit", self._save_profile), ("Uložit jako…", self._save_profile_as), ("Smazat", self._delete_profile)):
            b = QPushButton(text); b.clicked.connect(slot); prof.addWidget(b)
        prof.addStretch()
        lay.addLayout(prof)

        # režim
        mode = QGroupBox("Režim"); ml = QHBoxLayout(mode)
        self.mode_build = QRadioButton("Banka pro ithaca (analyze / build)")
        self.mode_slice = QRadioButton("Generický střih (slice)")
        self.mode_build.setChecked(True)
        self.mode_build.toggled.connect(self._update_mode)
        ml.addWidget(self.mode_build); ml.addWidget(self.mode_slice); ml.addStretch()
        lay.addWidget(mode)

        # složky
        folders = QGroupBox("Složky"); fl = QGridLayout(folders)
        self.dir_edits = {}
        for row, (key, label, hint) in enumerate((
                ("src", "Zdrojové nahrávky", "adresář se surovými WAV (např. 96 kHz / 24 bit); volitelně overrides.json"),
                ("original", "Original", "výstup 1: ořezané údery v původním formátu, m###/<hash>.wav + report.md + index"),
                ("bank", "Banka pro ithaca", "výstup 2: 48 kHz / 16 bit, tohle načítá ithaca"),
                ("out", "Výstup střihu", "generický střih: <zdroj>_slice_NNN_start_..ms_dur_..ms.wav"))):
            lab = QLabel(label + ":"); edit = QLineEdit(); edit.setPlaceholderText(hint); edit.setToolTip(hint)
            btn = QPushButton("…"); btn.setFixedWidth(36); btn.clicked.connect(lambda _, k=key: self._browse(k))
            fl.addWidget(lab, row, 0); fl.addWidget(edit, row, 1); fl.addWidget(btn, row, 2)
            self.dir_edits[key] = (lab, edit, btn)
        lay.addWidget(folders)

        # parametry
        params = QGroupBox("Parametry (ostatní prahy mají výchozí hodnoty ze specu; CLI --help je umí přepsat)")
        pl = QGridLayout(params)
        self.spins = {}
        for i, (key, label, lo, hi, step, unit, tip) in enumerate((
                ("end_level_db", "Konec dozvuku", -90, -20, 1, "dBFS", "úroveň, pod kterou surová data končí (efektivně max(hodnota, lokální dno + 6 dB)); pak umělý dozvuk do nuly"),
                ("tail_s", "Umělý dozvuk", 0.2, 10, 0.1, "s", "exponenciální doznění navazující na naměřený sklon"),
                ("max_len_s", "Max délka samplu", 1, 120, 1, "s", "horní limit délky surových dat"),
                ("preroll_ms", "Pre-roll", 0, 50, 1, "ms", "kolik vzít před nasazením (víc = latence při hraní)"),
                ("fade_in_ms", "Fade-in", 0, 20, 0.5, "ms", "proti kliku na začátku"))):
            sp = QDoubleSpinBox(); sp.setRange(lo, hi); sp.setSingleStep(step); sp.setSuffix(" " + unit)
            sp.setDecimals(1 if step < 1 else 0); sp.setToolTip(tip)
            pl.addWidget(QLabel(label + ":"), i // 3, (i % 3) * 2); pl.addWidget(sp, i // 3, (i % 3) * 2 + 1)
            self.spins[key] = sp
        self.retune = QCheckBox("Doladit na temperované ladění při převodu (--retune)")
        self.retune.setToolTip("posune výšku každého úderu o naměřenou odchylku; výchozí vypnuto (piano se ladí před samplováním)")
        pl.addWidget(self.retune, 2, 0, 1, 6)
        lay.addWidget(params)

        # akce
        act = QHBoxLayout()
        self.btn_analyze = QPushButton("Analyzovat (dry-run)"); self.btn_analyze.clicked.connect(lambda: self._run("analyze"))
        self.btn_build = QPushButton("Sestavit banku"); self.btn_build.clicked.connect(lambda: self._run("build"))
        self.btn_slice = QPushButton("Rozřezat"); self.btn_slice.clicked.connect(lambda: self._run("slice"))
        self.btn_report = QPushButton("Otevřít report"); self.btn_report.clicked.connect(self._open_report)
        for b in (self.btn_analyze, self.btn_build, self.btn_slice, self.btn_report):
            act.addWidget(b)
        act.addStretch()
        self.status = QLabel(""); act.addWidget(self.status)
        lay.addLayout(act)

        # log
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumBlockCount(5000)
        self.log.setFont(QFont("Menlo" if sys.platform == "darwin" else "Monospace", 11))
        lay.addWidget(self.log, 1)
        self._update_mode()

    # ---- profily -----------------------------------------------------------
    def params(self) -> dict:
        p = dict(DEFAULT_PARAMS)
        p["mode"] = "build" if self.mode_build.isChecked() else "slice"
        for k, (_, edit, _) in self.dir_edits.items():
            p[k] = edit.text().strip()
        for k, sp in self.spins.items():
            p[k] = sp.value()
        p["retune"] = self.retune.isChecked()
        return p

    def set_params(self, p: dict):
        (self.mode_build if p["mode"] == "build" else self.mode_slice).setChecked(True)
        for k, (_, edit, _) in self.dir_edits.items():
            edit.setText(p.get(k, ""))
        for k, sp in self.spins.items():
            sp.setValue(float(p[k]))
        self.retune.setChecked(bool(p["retune"]))
        self._update_mode()

    def _load_profile(self, name):
        if name:
            self.set_params(self.store.get(name))
            self.profile_box.blockSignals(True); self.profile_box.setCurrentText(name); self.profile_box.blockSignals(False)
        else:
            self.set_params(dict(DEFAULT_PARAMS))

    def _save_profile(self):
        name = self.profile_box.currentText().strip()
        if not name:
            return self._save_profile_as()
        self.store.put(name, self.params()); self._refresh_profiles(name)
        self._append(f"Profil „{name}“ uložen.", "INFO")

    def _save_profile_as(self):
        name, ok = QInputDialog.getText(self, "Uložit profil", "Název profilu (např. název banky):")
        if ok and name.strip():
            self.store.put(name.strip(), self.params()); self._refresh_profiles(name.strip())
            self._append(f"Profil „{name.strip()}“ uložen.", "INFO")

    def _delete_profile(self):
        name = self.profile_box.currentText()
        if name and QMessageBox.question(self, "Smazat profil", f"Smazat profil „{name}“?") == QMessageBox.Yes:
            self.store.delete(name); self._refresh_profiles(None)

    def _refresh_profiles(self, current):
        self.profile_box.blockSignals(True)
        self.profile_box.clear(); self.profile_box.addItems(self.store.names())
        if current:
            self.profile_box.setCurrentText(current)
        self.profile_box.blockSignals(False)

    # ---- chování -----------------------------------------------------------
    def _update_mode(self):
        build = self.mode_build.isChecked()
        for key, (lab, edit, btn) in self.dir_edits.items():
            visible = key == "src" or (build and key in ("original", "bank")) or (not build and key == "out")
            lab.setVisible(visible); edit.setVisible(visible); btn.setVisible(visible)
        self.retune.setVisible(build)
        self.spins["fade_in_ms"].setEnabled(True)
        for b, vis in ((self.btn_analyze, build), (self.btn_build, build), (self.btn_report, build), (self.btn_slice, not build)):
            b.setVisible(vis)

    def _browse(self, key):
        _, edit, _ = self.dir_edits[key]
        d = QFileDialog.getExistingDirectory(self, "Vyber adresář", edit.text() or str(Path.home()))
        if d:
            edit.setText(d)

    def _run(self, job: str):
        p = self.params()
        need = {"analyze": ["src"], "build": ["src", "original", "bank"], "slice": ["src", "out"]}[job]
        missing = [k for k in need if not p[k]]
        if missing:
            QMessageBox.warning(self, "Chybí složka", "Vyplň: " + ", ".join(self.dir_edits[k][0].text().rstrip(":") for k in missing))
            return
        if not Path(p["src"]).is_dir():
            QMessageBox.warning(self, "Chybí složka", f"Zdrojový adresář neexistuje: {p['src']}")
            return
        self._set_busy(True, {"analyze": "Analyzuji…", "build": "Stavím banku…", "slice": "Řežu…"}[job])
        self._append(f"== {job}: {p['src']}", "INFO")
        self.worker = JobWorker(job, p, self)
        self.worker.log.connect(self._append)
        self.worker.done.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_done(self, summary: str):
        self._append(summary, "INFO"); self._set_busy(False, "Hotovo")

    def _on_failed(self, msg: str):
        self._append(msg, "ERROR"); self._set_busy(False, "Chyba")

    def _set_busy(self, busy: bool, text: str):
        for b in (self.btn_analyze, self.btn_build, self.btn_slice):
            b.setEnabled(not busy)
        self.status.setText(text)

    def _append(self, msg: str, level: str = "INFO"):
        prefix = {"INFO": "", "TABLE": "", "WARN": "VAROVÁNÍ: ", "WARNING": "VAROVÁNÍ: ", "ERROR": "CHYBA: "}.get(level, "")
        self.log.appendPlainText(prefix + msg)

    def _open_report(self):
        rep = Path(self.params()["original"]) / "report.md" if self.params()["original"] else None
        if not rep or not rep.exists():
            QMessageBox.information(self, "Report", "Report ještě neexistuje — nejdřív sestav banku.")
            return
        opener = {"darwin": ["open"], "win32": ["cmd", "/c", "start", ""]}.get(sys.platform, ["xdg-open"])
        subprocess.Popen(opener + [str(rep)])

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "Běží úloha", "Počkej, až úloha doběhne.")
            event.ignore(); return
        event.accept()
