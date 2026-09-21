"""`python -m slicergui` / `sample-slicer-gui`: Qt okno nad sample_slicer (pip install -e '.[gui]')."""
import sys


def main() -> int:
    from PySide6.QtWidgets import QApplication
    from slicergui.app import MainWindow
    app = QApplication(sys.argv)
    app.setApplicationName("sample-slicer")
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
