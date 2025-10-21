# Audio Sample Slicer GUI

Grafické rozhraní pro zpracování audio vzorků s session managementem a hash-based caching.

## 📦 Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

**Requirements:**
- Python 3.8+
- PySide6 (Qt6 GUI framework)
- platformdirs (OS-specific paths)
- numpy, tqdm (audio processing)

---

## 🚀 Usage

### Launch GUI Application

```bash
python slicergui.py
```

### First Start

1. **Session Dialog** appears:
   - **Left panel**: Select existing session
   - **Right panel**: Create new session

2. Enter session name and click **Create** or select existing and click **Load**

3. **Main Window** opens with your session parameters

---

## 🎛️ Main Window Features

### 1. Folders
- **Input Directory**: WAV files to process
- **Output Directory**: Where sliced segments are saved

### 2. Detection Parameters
- **Threshold (dB)**: Audio level threshold for segment detection (-60 to 0)
- **Min Segment Length**: Minimum duration for detected segments (0.1-10s)
- **Min Length After Trim**: Minimum after silence removal (0.1-5s)
- **Trim Threshold Offset**: Additional dB for trimming (+0 to +20)

### 3. Processing Options
- **Fade Length**: Fade-in/out duration to prevent clicks (0-50ms)
- **Apply Fades**: Enable/disable fade processing
- **Resume**: Skip already processed files
- **Preview Mode**: Analyze only, no files created
- **Log Level**: DEBUG, INFO, WARNING, ERROR

### 4. Controls
- **START PROCESSING**: Begin processing WAV files
- **STOP**: Cancel current processing
- **Progress Bars**:
  - Total progress across all files
  - Current file progress

### 5. Log Output
- Real-time processing logs
- Color-coded by severity
- Scrollable with auto-cleanup

---

## 💾 Session Management

### What is a Session?

Sessions store:
- Input/output folder paths
- All processing parameters
- File processing cache (MD5 hashes)
- UI preferences

### Session Persistence

**Windows:**
```
C:\Users\{user}\AppData\Local\LordAudio\AudioSlicerSessions\
├── session-MyProject.json
├── session-MyProject.json.backup
└── ...
```

**macOS:**
```
~/Library/Application Support/LordAudio/AudioSlicerSessions/
```

**Linux:**
```
~/.local/share/LordAudio/AudioSlicerSessions/
```

### Session Operations

- **Switch Session**: Change to another session
- **Save Session**: Manually save (auto-save enabled by default)
- **Delete Session**: Remove session from Session Dialog

---

## 🔍 Hash-Based Caching

The application tracks processed files using MD5 hashes:

- **First processing**: File hash calculated and stored
- **Re-processing**: Hash compared to detect file changes
- **Cache data**: Filename, format, segments created, timestamp

**Benefits:**
- Faster re-processing detection
- Track processing history
- Identify duplicate files

---

## 📁 Project Structure

```
sample-slicer/
├── slicer.py                    # Original CLI tool (unchanged)
├── slicergui.py                 # GUI entry point
├── requirements.txt             # Dependencies
│
└── slicergui/
    ├── config/
    │   ├── __init__.py          # SESSIONS_DIR (platformdirs)
    │   └── app_config.py        # Constants and defaults
    │
    ├── domain/
    │   └── interfaces/
    │       └── session_repository.py  # Repository interface (ABC)
    │
    ├── infrastructure/
    │   └── persistence/
    │       └── session_repository_impl.py  # JSON implementation
    │
    ├── logic.py                 # Audio processing core
    ├── session_manager.py       # Session CRUD + hash cache
    ├── worker.py                # QThread for async processing
    └── gui.py                   # PySide6 UI components
```

---

## 🎯 Workflow Example

1. **Start application**
   ```bash
   python slicergui.py
   ```

2. **Create session** "DrumSamples"

3. **Configure**:
   - Input: `C:/audio/drums_raw/`
   - Output: `C:/audio/drums_sliced/`
   - Threshold: -45 dB
   - Min Length: 3.0s

4. **Start Processing**:
   - 5 WAV files found
   - Real-time progress shown
   - Logs display detection info

5. **Results**:
   - 23 segments created
   - Session auto-saved with cache

6. **Next time**:
   - Load "DrumSamples" session
   - All parameters restored
   - Add more files, resume processing

---

## 🛠️ CLI Tool (Original)

The original CLI `slicer.py` remains **unchanged** and fully functional:

```bash
python slicer.py \
  --input-dir samples_in \
  --output-dir samples_out \
  --threshold_db -45 \
  --min_length 3
```

See `slicer.py --help` for all CLI options.

---

## 🔧 Architecture Highlights

### Clean Architecture Pattern
- **Domain Layer**: Business logic interfaces
- **Application Layer**: Session manager, use cases
- **Infrastructure Layer**: Persistence (JSON with backup)
- **Presentation Layer**: PySide6 GUI

### Repository Pattern
- Abstract `ISessionRepository` interface
- Concrete `JsonSessionRepository` implementation
- Easy to swap backends (SQL, Redis, etc.)

### Threading
- `QThread` worker for async processing
- GUI remains responsive during operations
- Signals for progress updates

### Auto-Save
- Parameters auto-saved after 500ms inactivity
- Session saved on window close
- Backup mechanism prevents data loss

---

## 📝 Logging

Logs are saved to:
```
~/.audioslicer/slicergui.log
```

Console also shows INFO level messages.

---

## 🤝 Contributing

- Original CLI: `slicer.py`
- GUI Extension: `slicergui/` package
- Both tools share core processing logic

---

## 📄 License

Part of the LordAudio sample-slicer project.

---

**Enjoy slicing! 🎵**
