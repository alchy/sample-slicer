"""
GUI Components
==============
PySide6 GUI pro Audio Sample Slicer.
Obsahuje SessionDialog a MainWindow.
"""

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QLineEdit, QPushButton, QSlider, QCheckBox, QComboBox,
    QProgressBar, QTextEdit, QFileDialog, QMessageBox,
    QDialog, QListWidget, QDialogButtonBox, QInputDialog
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont

from slicergui.config.app_config import ProcessingDefaults, GUIConfig, SessionConfig
from slicergui.session_manager import SessionManager
from slicergui.worker import ProcessingWorker

logger = logging.getLogger(__name__)


# ============================================================================
# SESSION DIALOG
# ============================================================================

class SessionDialog(QDialog):
    """
    Dialog pro výběr existující session nebo vytvoření nové.
    Layout: vlevo seznam sessions, vpravo Create New.
    """

    def __init__(self, session_manager: SessionManager, parent=None):
        super().__init__(parent)

        self.session_manager = session_manager
        self.selected_session: Optional[str] = None

        self.setWindowTitle("Select Session")
        self.setModal(True)
        self.resize(GUIConfig.SESSION_DIALOG_WIDTH, GUIConfig.SESSION_DIALOG_HEIGHT)

        self._setup_ui()
        self._load_sessions()

    def _setup_ui(self):
        """Vytvoření UI dialogu"""
        layout = QVBoxLayout(self)

        # Popis
        description = QLabel("Select an existing session or create a new one:")
        layout.addWidget(description)

        # Hlavní horizontální layout
        main_layout = QHBoxLayout()

        # --- Levá strana: Seznam sessions ---
        left_group = QGroupBox("Existing Sessions")
        left_layout = QVBoxLayout()

        self.sessions_list = QListWidget()
        self.sessions_list.itemDoubleClicked.connect(self._on_session_double_click)
        left_layout.addWidget(self.sessions_list)

        # Tlačítka pro sessions
        btn_layout = QHBoxLayout()
        self.load_btn = QPushButton("Load Selected")
        self.load_btn.clicked.connect(self._on_load_clicked)
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.clicked.connect(self._on_delete_clicked)

        btn_layout.addWidget(self.load_btn)
        btn_layout.addWidget(self.delete_btn)
        left_layout.addLayout(btn_layout)

        left_group.setLayout(left_layout)
        main_layout.addWidget(left_group, 2)  # 2/3 šířky

        # --- Pravá strana: Create New ---
        right_group = QGroupBox("Create New Session")
        right_layout = QVBoxLayout()

        right_layout.addWidget(QLabel("Session Name:"))
        self.new_session_input = QLineEdit()
        self.new_session_input.setPlaceholderText("Enter session name...")
        self.new_session_input.returnPressed.connect(self._on_create_clicked)
        right_layout.addWidget(self.new_session_input)

        self.create_btn = QPushButton("Create")
        self.create_btn.clicked.connect(self._on_create_clicked)
        right_layout.addWidget(self.create_btn)

        right_layout.addStretch()
        right_group.setLayout(right_layout)
        main_layout.addWidget(right_group, 1)  # 1/3 šířky

        layout.addLayout(main_layout)

        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Cancel)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _load_sessions(self):
        """Načte seznam sessions do list widgetu"""
        self.sessions_list.clear()
        sessions = self.session_manager.list_all_sessions()

        if not sessions:
            self.sessions_list.addItem("(No sessions found)")
            self.load_btn.setEnabled(False)
            self.delete_btn.setEnabled(False)
        else:
            self.sessions_list.addItems(sessions)
            self.load_btn.setEnabled(True)
            self.delete_btn.setEnabled(True)

    def _on_session_double_click(self, item):
        """Double-click na session = load"""
        if item.text() != "(No sessions found)":
            self.selected_session = item.text()
            self.accept()

    def _on_load_clicked(self):
        """Load vybrané session"""
        current_item = self.sessions_list.currentItem()
        if current_item and current_item.text() != "(No sessions found)":
            self.selected_session = current_item.text()
            self.accept()
        else:
            QMessageBox.warning(self, "No Selection", "Please select a session to load.")

    def _on_delete_clicked(self):
        """Smazání vybrané session"""
        current_item = self.sessions_list.currentItem()
        if not current_item or current_item.text() == "(No sessions found)":
            QMessageBox.warning(self, "No Selection", "Please select a session to delete.")
            return

        session_name = current_item.text()

        # Potvrzení
        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete session '{session_name}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            if self.session_manager.delete_session(session_name):
                QMessageBox.information(self, "Success", f"Session '{session_name}' deleted.")
                self._load_sessions()
            else:
                QMessageBox.critical(self, "Error", f"Failed to delete session '{session_name}'.")

    def _on_create_clicked(self):
        """Vytvoření nové session"""
        session_name = self.new_session_input.text().strip()

        if not session_name:
            QMessageBox.warning(self, "Invalid Name", "Session name cannot be empty.")
            return

        # Validace názvu
        if len(session_name) < SessionConfig.MIN_SESSION_NAME_LENGTH:
            QMessageBox.warning(self, "Invalid Name", "Session name is too short.")
            return

        if len(session_name) > SessionConfig.MAX_SESSION_NAME_LENGTH:
            QMessageBox.warning(self, "Invalid Name", "Session name is too long.")
            return

        for char in SessionConfig.INVALID_CHARS:
            if char in session_name:
                QMessageBox.warning(
                    self,
                    "Invalid Name",
                    f"Session name contains invalid character: '{char}'"
                )
                return

        # Kontrola existence
        if self.session_manager.repository.exists(session_name):
            QMessageBox.warning(self, "Already Exists", f"Session '{session_name}' already exists.")
            return

        # Vytvoření
        if self.session_manager.create_session(session_name):
            self.selected_session = session_name
            QMessageBox.information(self, "Success", f"Session '{session_name}' created.")
            self.accept()
        else:
            QMessageBox.critical(self, "Error", f"Failed to create session '{session_name}'.")

    def get_selected_session(self) -> Optional[str]:
        """Vrátí vybranou/vytvořenou session"""
        return self.selected_session


# ============================================================================
# MAIN WINDOW
# ============================================================================

class MainWindow(QMainWindow):
    """
    Hlavní okno aplikace s parametry, controls a progress.
    """

    def __init__(self, session_manager: SessionManager):
        super().__init__()

        self.session_manager = session_manager
        self.worker: Optional[ProcessingWorker] = None

        self.setWindowTitle("Audio Sample Slicer")
        self.resize(GUIConfig.WINDOW_WIDTH, GUIConfig.WINDOW_HEIGHT)
        self.setMinimumSize(GUIConfig.WINDOW_MIN_WIDTH, GUIConfig.WINDOW_MIN_HEIGHT)

        self._setup_ui()
        self._load_session_data()
        self._connect_signals()

    def _setup_ui(self):
        """Vytvoření kompletního UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        # Session header
        main_layout.addWidget(self._create_session_header())

        # Folders
        main_layout.addWidget(self._create_folders_group())

        # Detection parameters
        main_layout.addWidget(self._create_detection_group())

        # Processing options
        main_layout.addWidget(self._create_processing_group())

        # Controls
        main_layout.addWidget(self._create_controls_group())

        # Log output
        main_layout.addWidget(self._create_log_group())

        # Status bar
        self.statusBar().showMessage("Ready")

    def _create_session_header(self) -> QWidget:
        """Header s názvem session a tlačítky"""
        widget = QWidget()
        layout = QHBoxLayout(widget)

        # Session info
        session_info = self.session_manager.get_session_info()
        session_name = session_info.get("session_name", "Unknown")

        label = QLabel(f"<b>Session:</b> {session_name}")
        label.setFont(QFont("", 10, QFont.Bold))
        layout.addWidget(label)

        layout.addStretch()

        # Tlačítka
        switch_btn = QPushButton("Switch Session")
        switch_btn.clicked.connect(self._on_switch_session)
        layout.addWidget(switch_btn)

        save_btn = QPushButton("Save Session")
        save_btn.clicked.connect(self._on_save_session)
        layout.addWidget(save_btn)

        return widget

    def _create_folders_group(self) -> QGroupBox:
        """Skupina pro input/output složky"""
        group = QGroupBox("Folders")
        layout = QVBoxLayout()

        # Input folder
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Input Directory:"))
        self.input_dir_edit = QLineEdit()
        self.input_dir_edit.setPlaceholderText("Select input folder...")
        input_layout.addWidget(self.input_dir_edit)

        input_btn = QPushButton("Browse...")
        input_btn.clicked.connect(self._on_browse_input)
        input_layout.addWidget(input_btn)

        layout.addLayout(input_layout)

        # Output folder
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output Directory:"))
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setPlaceholderText("Select output folder...")
        output_layout.addWidget(self.output_dir_edit)

        output_btn = QPushButton("Browse...")
        output_btn.clicked.connect(self._on_browse_output)
        output_layout.addWidget(output_btn)

        layout.addLayout(output_layout)

        group.setLayout(layout)
        return group

    def _create_detection_group(self) -> QGroupBox:
        """Skupina pro detekční parametry se slidery"""
        group = QGroupBox("Detection Parameters")
        layout = QVBoxLayout()

        # Threshold dB
        layout.addWidget(self._create_slider_row(
            "Threshold (dB):",
            ProcessingDefaults.THRESHOLD_MIN,
            ProcessingDefaults.THRESHOLD_MAX,
            ProcessingDefaults.THRESHOLD_DB,
            GUIConfig.SLIDER_STEPS_THRESHOLD,
            "threshold_slider",
            "threshold_label"
        ))

        # Min Length
        layout.addWidget(self._create_slider_row(
            "Min Segment Length (s):",
            ProcessingDefaults.MIN_LENGTH_MIN,
            ProcessingDefaults.MIN_LENGTH_MAX,
            ProcessingDefaults.MIN_LENGTH,
            GUIConfig.SLIDER_STEPS_LENGTH,
            "min_length_slider",
            "min_length_label"
        ))

        # Min Length After Trim
        layout.addWidget(self._create_slider_row(
            "Min Length After Trim (s):",
            ProcessingDefaults.MIN_LENGTH_AFTER_TRIM_MIN,
            ProcessingDefaults.MIN_LENGTH_AFTER_TRIM_MAX,
            ProcessingDefaults.MIN_LENGTH_AFTER_TRIM,
            GUIConfig.SLIDER_STEPS_TRIM,
            "min_trim_slider",
            "min_trim_label"
        ))

        # Trim Offset
        layout.addWidget(self._create_slider_row(
            "Trim Threshold Offset (dB):",
            ProcessingDefaults.TRIM_THRESHOLD_OFFSET_MIN,
            ProcessingDefaults.TRIM_THRESHOLD_OFFSET_MAX,
            ProcessingDefaults.TRIM_THRESHOLD_OFFSET,
            GUIConfig.SLIDER_STEPS_OFFSET,
            "trim_offset_slider",
            "trim_offset_label"
        ))

        group.setLayout(layout)
        return group

    def _create_processing_group(self) -> QGroupBox:
        """Skupina pro processing options"""
        group = QGroupBox("Processing Options")
        layout = QVBoxLayout()

        # Fade-in (ms)
        layout.addWidget(self._create_slider_row(
            "Fade-In Length (ms):",
            ProcessingDefaults.FADE_IN_MS_MIN,
            ProcessingDefaults.FADE_IN_MS_MAX,
            ProcessingDefaults.FADE_IN_MS,
            GUIConfig.SLIDER_STEPS_FADE_IN,
            "fade_in_slider",
            "fade_in_label"
        ))

        # Fade-out (%)
        layout.addWidget(self._create_slider_row(
            "Fade-Out Length (% of segment):",
            ProcessingDefaults.FADE_OUT_PERCENT_MIN,
            ProcessingDefaults.FADE_OUT_PERCENT_MAX,
            ProcessingDefaults.FADE_OUT_PERCENT,
            GUIConfig.SLIDER_STEPS_FADE_OUT,
            "fade_out_slider",
            "fade_out_label"
        ))

        # Checkboxy
        checks_layout = QHBoxLayout()

        self.apply_fades_check = QCheckBox("Apply Fades")
        self.apply_fades_check.setChecked(ProcessingDefaults.APPLY_FADES)
        checks_layout.addWidget(self.apply_fades_check)

        self.overwrite_check = QCheckBox("Overwrite Files")
        self.overwrite_check.setChecked(ProcessingDefaults.OVERWRITE)
        checks_layout.addWidget(self.overwrite_check)

        self.resume_check = QCheckBox("Resume (skip existing)")
        self.resume_check.setChecked(ProcessingDefaults.RESUME)
        checks_layout.addWidget(self.resume_check)

        self.preview_check = QCheckBox("Preview Mode")
        self.preview_check.setChecked(ProcessingDefaults.PREVIEW)
        checks_layout.addWidget(self.preview_check)

        checks_layout.addStretch()
        layout.addLayout(checks_layout)

        # Log level
        log_layout = QHBoxLayout()
        log_layout.addWidget(QLabel("Log Level:"))
        self.log_level_combo = QComboBox()
        self.log_level_combo.addItems(ProcessingDefaults.LOG_LEVELS)
        self.log_level_combo.setCurrentText(ProcessingDefaults.LOG_LEVEL)
        log_layout.addWidget(self.log_level_combo)
        log_layout.addStretch()

        layout.addLayout(log_layout)

        group.setLayout(layout)
        return group

    def _create_controls_group(self) -> QGroupBox:
        """Skupina s tlačítky a progress bars"""
        group = QGroupBox("Controls")
        layout = QVBoxLayout()

        # Tlačítka
        btn_layout = QHBoxLayout()

        self.start_btn = QPushButton("▶ START PROCESSING")
        self.start_btn.clicked.connect(self._on_start_processing)
        self.start_btn.setMinimumHeight(40)
        btn_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("■ STOP")
        self.stop_btn.clicked.connect(self._on_stop_processing)
        self.stop_btn.setEnabled(False)
        self.stop_btn.setMinimumHeight(40)
        btn_layout.addWidget(self.stop_btn)

        layout.addLayout(btn_layout)

        # Progress bars
        layout.addWidget(QLabel("Total Progress:"))
        self.total_progress = QProgressBar()
        self.total_progress.setValue(0)
        layout.addWidget(self.total_progress)

        self.current_file_label = QLabel("Current File: -")
        layout.addWidget(self.current_file_label)

        self.file_progress = QProgressBar()
        self.file_progress.setValue(0)
        layout.addWidget(self.file_progress)

        group.setLayout(layout)
        return group

    def _create_log_group(self) -> QGroupBox:
        """Skupina pro log output"""
        group = QGroupBox("Log Output")
        layout = QVBoxLayout()

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        layout.addWidget(self.log_text)

        # Clear button
        clear_btn = QPushButton("Clear Log")
        clear_btn.clicked.connect(self.log_text.clear)
        layout.addWidget(clear_btn)

        group.setLayout(layout)
        return group

    def _create_slider_row(
        self,
        label_text: str,
        min_val: float,
        max_val: float,
        default_val: float,
        steps: int,
        slider_attr: str,
        label_attr: str
    ) -> QWidget:
        """
        Vytvoří řádek se sliderem a labelem hodnoty.

        Args:
            label_text: Text labelu
            min_val: Minimální hodnota
            max_val: Maximální hodnota
            default_val: Výchozí hodnota
            steps: Počet kroků slideru
            slider_attr: Název atributu pro slider (self.{slider_attr})
            label_attr: Název atributu pro label (self.{label_attr})
        """
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        # Label
        layout.addWidget(QLabel(label_text))

        # Slider
        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(steps)

        # Převod default hodnoty na slider pozici
        normalized = (default_val - min_val) / (max_val - min_val)
        slider.setValue(int(normalized * steps))

        # Value label
        value_label = QLabel(f"{default_val:.1f}")
        value_label.setMinimumWidth(50)

        # Connect slider k update labelu
        def update_label(pos):
            # Převod slider pozice na skutečnou hodnotu
            norm = pos / steps
            value = min_val + norm * (max_val - min_val)
            value_label.setText(f"{value:.1f}")

        slider.valueChanged.connect(update_label)

        layout.addWidget(slider, 3)
        layout.addWidget(value_label)

        # Uložení referencí jako atributy
        setattr(self, slider_attr, slider)
        setattr(self, label_attr, value_label)

        # Uložení min/max pro pozdější konverzi
        slider.setProperty("min_val", min_val)
        slider.setProperty("max_val", max_val)

        return widget

    def _slider_to_value(self, slider: QSlider) -> float:
        """Převede slider pozici na skutečnou hodnotu"""
        min_val = slider.property("min_val")
        max_val = slider.property("max_val")
        steps = slider.maximum()
        pos = slider.value()

        normalized = pos / steps
        return min_val + normalized * (max_val - min_val)

    def _value_to_slider(self, slider: QSlider, value: float):
        """Nastaví slider hodnotu z float"""
        min_val = slider.property("min_val")
        max_val = slider.property("max_val")
        steps = slider.maximum()

        normalized = (value - min_val) / (max_val - min_val)
        slider.setValue(int(normalized * steps))

    def _connect_signals(self):
        """Připojení signálů pro auto-save"""
        # Složky
        self.input_dir_edit.textChanged.connect(self._on_params_changed)
        self.output_dir_edit.textChanged.connect(self._on_params_changed)

        # Slidery
        self.threshold_slider.valueChanged.connect(self._on_params_changed)
        self.min_length_slider.valueChanged.connect(self._on_params_changed)
        self.min_trim_slider.valueChanged.connect(self._on_params_changed)
        self.trim_offset_slider.valueChanged.connect(self._on_params_changed)
        self.fade_in_slider.valueChanged.connect(self._on_params_changed)
        self.fade_out_slider.valueChanged.connect(self._on_params_changed)

        # Checkboxy
        self.apply_fades_check.stateChanged.connect(self._on_params_changed)
        self.overwrite_check.stateChanged.connect(self._on_params_changed)
        self.resume_check.stateChanged.connect(self._on_params_changed)
        self.preview_check.stateChanged.connect(self._on_params_changed)

        # Combo
        self.log_level_combo.currentTextChanged.connect(self._on_params_changed)

    def _load_session_data(self):
        """Načte data z session do GUI"""
        if not self.session_manager.is_session_loaded():
            return

        # Složky
        folders = self.session_manager.get_folders()
        self.input_dir_edit.setText(folders.get("input") or "")
        self.output_dir_edit.setText(folders.get("output") or "")

        # Parametry
        params = self.session_manager.get_processing_params()

        self._value_to_slider(self.threshold_slider, params.get("threshold_db", ProcessingDefaults.THRESHOLD_DB))
        self._value_to_slider(self.min_length_slider, params.get("min_length", ProcessingDefaults.MIN_LENGTH))
        self._value_to_slider(self.min_trim_slider, params.get("min_length_after_trim", ProcessingDefaults.MIN_LENGTH_AFTER_TRIM))
        self._value_to_slider(self.trim_offset_slider, params.get("trim_threshold_offset", ProcessingDefaults.TRIM_THRESHOLD_OFFSET))
        self._value_to_slider(self.fade_in_slider, params.get("fade_in_ms", ProcessingDefaults.FADE_IN_MS))
        self._value_to_slider(self.fade_out_slider, params.get("fade_out_percent", ProcessingDefaults.FADE_OUT_PERCENT))

        self.apply_fades_check.setChecked(params.get("apply_fades", ProcessingDefaults.APPLY_FADES))
        self.overwrite_check.setChecked(params.get("overwrite", ProcessingDefaults.OVERWRITE))
        self.resume_check.setChecked(params.get("resume", ProcessingDefaults.RESUME))
        self.preview_check.setChecked(params.get("preview", ProcessingDefaults.PREVIEW))

        self.log_level_combo.setCurrentText(params.get("log_level", ProcessingDefaults.LOG_LEVEL))

        self._append_log("Session loaded successfully", "INFO")

    def _on_params_changed(self):
        """Auto-save při změně parametrů (s debounce)"""
        # Použijeme QTimer pro debounce (uložíme až po 500ms neaktivity)
        if not hasattr(self, '_save_timer'):
            self._save_timer = QTimer()
            self._save_timer.setSingleShot(True)
            self._save_timer.timeout.connect(self._save_params_to_session)

        self._save_timer.start(500)

    def _save_params_to_session(self):
        """Uloží aktuální parametry do session"""
        if not self.session_manager.is_session_loaded():
            return

        # Složky
        self.session_manager.set_folders(
            input_dir=self.input_dir_edit.text() or None,
            output_dir=self.output_dir_edit.text() or None
        )

        # Parametry
        self.session_manager.update_processing_params(
            threshold_db=self._slider_to_value(self.threshold_slider),
            min_length=self._slider_to_value(self.min_length_slider),
            min_length_after_trim=self._slider_to_value(self.min_trim_slider),
            trim_threshold_offset=self._slider_to_value(self.trim_offset_slider),
            fade_in_ms=self._slider_to_value(self.fade_in_slider),
            fade_out_percent=self._slider_to_value(self.fade_out_slider),
            apply_fades=self.apply_fades_check.isChecked(),
            overwrite=self.overwrite_check.isChecked(),
            resume=self.resume_check.isChecked(),
            preview=self.preview_check.isChecked(),
            log_level=self.log_level_combo.currentText()
        )

    def _on_browse_input(self):
        """Browse input directory"""
        current = self.input_dir_edit.text()
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Input Directory",
            current or ""
        )
        if folder:
            self.input_dir_edit.setText(folder)

    def _on_browse_output(self):
        """Browse output directory"""
        current = self.output_dir_edit.text()
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            current or ""
        )
        if folder:
            self.output_dir_edit.setText(folder)

    def _on_switch_session(self):
        """Přepnutí na jinou session"""
        dialog = SessionDialog(self.session_manager, self)
        if dialog.exec() == QDialog.Accepted:
            selected = dialog.get_selected_session()
            if selected:
                if self.session_manager.load_session(selected):
                    self._load_session_data()
                    self.setWindowTitle(f"Audio Sample Slicer - {selected}")
                    QMessageBox.information(self, "Success", f"Switched to session: {selected}")
                else:
                    QMessageBox.critical(self, "Error", "Failed to load session")

    def _on_save_session(self):
        """Manuální uložení session"""
        self._save_params_to_session()
        if self.session_manager.save_session():
            self.statusBar().showMessage("Session saved", 2000)
        else:
            QMessageBox.warning(self, "Error", "Failed to save session")

    def _on_start_processing(self):
        """Zahájení zpracování"""
        # Validace
        input_dir = self.input_dir_edit.text()
        output_dir = self.output_dir_edit.text()

        if not input_dir or not Path(input_dir).exists():
            QMessageBox.warning(self, "Invalid Input", "Input directory does not exist.")
            return

        if not output_dir:
            QMessageBox.warning(self, "Invalid Output", "Please select an output directory.")
            return

        # Vytvoření output dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Najdi WAV soubory
        input_path = Path(input_dir)
        wav_files = list(input_path.glob("*.[wW][aA][vV]"))

        if not wav_files:
            QMessageBox.warning(self, "No Files", "No WAV files found in input directory.")
            return

        self._append_log(f"Found {len(wav_files)} WAV files to process", "INFO")

        # Připrav parametry
        params = {
            "threshold_db": self._slider_to_value(self.threshold_slider),
            "min_length": self._slider_to_value(self.min_length_slider),
            "min_length_after_trim": self._slider_to_value(self.min_trim_slider),
            "trim_threshold_offset": self._slider_to_value(self.trim_offset_slider),
            "fade_in_ms": self._slider_to_value(self.fade_in_slider),
            "fade_out_percent": self._slider_to_value(self.fade_out_slider),
            "apply_fades": self.apply_fades_check.isChecked(),
            "overwrite": self.overwrite_check.isChecked(),
            "resume": self.resume_check.isChecked(),
            "preview": self.preview_check.isChecked(),
            "log_level": self.log_level_combo.currentText()
        }

        # Vytvoř worker
        self.worker = ProcessingWorker(wav_files, output_dir, params)

        # Připoj signály
        self.worker.progress_total.connect(self.total_progress.setValue)
        self.worker.progress_file.connect(self.file_progress.setValue)
        self.worker.current_file.connect(lambda f: self.current_file_label.setText(f"Current File: {f}"))
        self.worker.log_message.connect(self._append_log)
        self.worker.finished.connect(self._on_processing_finished)
        self.worker.error.connect(self._on_processing_error)

        # UI update
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.statusBar().showMessage("Processing...")

        # Start worker
        self.worker.start()

    def _on_stop_processing(self):
        """Zastavení zpracování"""
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.stop_btn.setEnabled(False)

    def _on_processing_finished(self, stats: dict):
        """Callback po dokončení zpracování"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.statusBar().showMessage("Processing completed!")

        # Reset progress
        self.total_progress.setValue(100)
        self.file_progress.setValue(100)
        self.current_file_label.setText("Current File: -")

        # Dialog s výsledky
        QMessageBox.information(
            self,
            "Processing Complete",
            f"Processing finished!\n\n"
            f"Files processed: {stats.get('files_processed', 0)}\n"
            f"Files failed: {stats.get('files_failed', 0)}\n"
            f"Segments created: {stats.get('segments_created', 0)}\n"
            f"Segments skipped: {stats.get('segments_skipped', 0)}"
        )

    def _on_processing_error(self, error_msg: str):
        """Callback při chybě"""
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.statusBar().showMessage("Error!")

        QMessageBox.critical(self, "Processing Error", error_msg)

    def _append_log(self, message: str, level: str = "INFO"):
        """Přidá zprávu do logu"""
        color_map = {
            "DEBUG": "gray",
            "INFO": "black",
            "WARNING": "orange",
            "ERROR": "red"
        }

        color = color_map.get(level, "black")
        self.log_text.append(f"<span style='color:{color}'><b>[{level}]</b> {message}</span>")

        # Omezení počtu řádků
        doc = self.log_text.document()
        if doc.blockCount() > GUIConfig.LOG_MAX_LINES:
            cursor = self.log_text.textCursor()
            cursor.movePosition(cursor.Start)
            cursor.select(cursor.BlockUnderCursor)
            cursor.removeSelectedText()
            cursor.deletePreviousChar()

    def closeEvent(self, event):
        """Override closeEvent pro uložení session před zavřením"""
        self._save_params_to_session()
        self.session_manager.save_session()
        event.accept()
