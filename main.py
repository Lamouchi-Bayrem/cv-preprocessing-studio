# main.py

import sys
from PyQt6.QtWidgets import QApplication
from gui.qt_app import CVStudio

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("CV Studio - Advanced CV Preprocessing & Augmentation")

    window = CVStudio()
    window.show()

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
