from qdarktheme import load_stylesheet
from sys import exit as sys_exit

from PySide6.QtWidgets import QApplication


def main():
    app = QApplication([])
    app.setStyleSheet(load_stylesheet())

    show_main_window()

    sys_exit(app.exec())


def show_main_window():
    from gui import MainWindow

    window = MainWindow()
    window.show()


if __name__ == "__main__":
    main()
