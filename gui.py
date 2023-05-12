from PySide6.QtCore import QSize
from PySide6.QtWidgets import QMainWindow, QPushButton, QFileDialog, QLabel
from news_analysis import get_text_file_content, get_analyzation_result


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Fake News Detector")

        self.submit_button = QPushButton("Submit", self)
        self.submit_button.move(10, 10)
        self.submit_button.clicked.connect(self.analyze_file)

        self.file_dialog_button = QPushButton("Select File", self)
        self.file_dialog_button.move(300, 10)
        self.file_dialog_button.clicked.connect(self.get_file_name)

        self.result_label = QLabel(self)
        self.result_label.setGeometry(10, 50, 400, 200)

        self.setFixedSize(QSize(800, 600))

    def get_file_name(self):
        file_filter = "Text files (*.txt)"
        response = QFileDialog.getOpenFileName(
            parent=self,
            caption="Select a file",
            filter=file_filter,
        )
        self.selected_file_path = response[0]

    def analyze_file(self):
        raw_user_article_text = get_text_file_content(self.selected_file_path)
        result = get_analyzation_result(raw_user_article_text)
        self.result_label.setText(
            f"Sentiment: {result['sentiment']}\nLabel: {result['label']}"
        )
