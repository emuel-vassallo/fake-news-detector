from os import getcwd
from PySide6.QtCore import QSize
from PySide6.QtWidgets import QMainWindow, QPushButton, QFileDialog


# Subclass QMainWindow to customize your application's main window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Fake News Detector")

        submit_button = QPushButton("Submit", self)

        submit_button.move(10, 10)  # adjust the position as needed

        file_dialog_button = QPushButton("Open File", self)
        file_dialog_button.move(300, 10)  # adjust the position as needed
        file_dialog_button.clicked.connect(self.get_file_name)

        self.setFixedSize(QSize(800, 600))

    def get_file_name(self):
        file_filter = "Text files (*.txt)"
        response = QFileDialog.getOpenFileName(
            parent=self,
            caption="Select a file",
            filter=file_filter,
        )
