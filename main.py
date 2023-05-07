from qdarktheme import load_stylesheet
from sys import exit as sys_exit
from os import system

from PySide6.QtWidgets import QApplication, QMainWindow

from gui import MainWindow


def main():
    app = QApplication([])
    app.setStyleSheet(load_stylesheet())

    window = MainWindow()
    window.show()

    sys_exit(app.exec())


if __name__ == "__main__":
    main()
