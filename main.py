from PyQt6.QtWidgets import QApplication
import sys
from gui.qt_app import CVStudio

def main():
    app = QApplication(sys.argv)
    window = CVStudio()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

