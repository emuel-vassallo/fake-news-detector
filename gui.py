import os

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QMainWindow,
    QPushButton,
    QFileDialog,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
)
from news_analysis import (
    is_model_trained,
    get_trained_model,
    get_text_file_content,
    get_analyzation_result,
)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Fake News Detector")
        self.setFixedSize(QSize(800, 600))

        self.box_layout = QVBoxLayout()
        self.widget = QWidget(self)
        self.widget.setLayout(self.box_layout)

        self.title_label = QLabel(self)
        self.title_label.setText("Analyse Sentiment and Detect Fake News!")
        self.title_label.setFont(QFont("Arial", 21, QFont.Bold))
        self.box_layout.addWidget(self.title_label, alignment=Qt.AlignHCenter)

        self.box_layout.addSpacing(26)

        self.file_layout = QHBoxLayout()
        self.box_layout.addLayout(self.file_layout)

        self.selected_file_label = QLabel(self)
        self.selected_file_label.setMaximumWidth(180)
        self.selected_file_label.setText("Select Article Text File:")
        self.selected_file_label.setFont(QFont("Arial", 11, QFont.Bold))

        self.file_dialog_button = QPushButton("No file selected", self)
        self.file_dialog_button.setMaximumWidth(500)
        self.file_dialog_button.clicked.connect(self.get_file_name)
        self.file_dialog_button.setFont(QFont("Arial", 11))
        self.file_dialog_button.setStyleSheet(
            """
            QPushButton {
                background-color: transparent;
                color: #6c757d;
                border-color: #6c757d;
            }
            QPushButton:hover {
                color: #fff;
                background-color: #6c757d;
                border-color: #6c757d;
            }
            QPushButton:pressed {
                color: #fff;
                background-color: #6c757d;
                border-color: #6c757d;
            }
        """
        )
        self.file_dialog_button.setCursor(Qt.PointingHandCursor)

        self.file_layout.addWidget(self.selected_file_label, stretch=0)
        self.file_layout.addWidget(self.file_dialog_button, stretch=0)
        
        self.box_layout.addSpacing(26)

        self.submit_button = QPushButton("Start Analysing", self)
        self.submit_button.setFixedSize(160, 40)
        self.submit_button.setFont(QFont("Arial", 11, QFont.Bold))
        self.submit_button.setStyleSheet(
            """
            QPushButton {
                background-color: #007bff;
                color: #fff;
                border-color: #0062cc;
            }
            QPushButton:hover {
                background-color: #0069d9;
            }
            QPushButton:pressed {
                background-color: #0062cc;
            }
        """
        )
        self.submit_button.clicked.connect(self.analyze_file)
        self.submit_button.setCursor(Qt.PointingHandCursor)
        self.box_layout.addWidget(self.submit_button, alignment=Qt.AlignHCenter)

        self.result_label = QLabel(self)
        self.box_layout.addWidget(self.result_label, alignment=Qt.AlignHCenter)

        self.is_model_trained_label = QLabel(self)
        self.box_layout.addWidget(
            self.is_model_trained_label, alignment=Qt.AlignHCenter
        )

        self.box_layout.setAlignment(Qt.AlignCenter)

        self.setCentralWidget(self.widget)

        self.populate_id_model_trained_yet()

    def get_file_name(self):
        file_filter = "Text files (*.txt)"
        response = QFileDialog.getOpenFileName(
            parent=self,
            caption="Select a file",
            filter=file_filter,
        )
        self.selected_file_path = response[0]
        file_name = os.path.basename(self.selected_file_path)
        self.selected_file_name = file_name
        self.file_dialog_button.setText(file_name)

    def populate_id_model_trained_yet(self):
        is_model_trained_yet = is_model_trained()
        self.is_model_trained_label.setText(
            "The model is trained. Submit an article to analyse."
            if is_model_trained_yet
            else "The model will begin training once you submit an article."
        )
        
    def analyze_file(self):
        raw_user_article_text = get_text_file_content(self.selected_file_path)
        vectorizer, clf = get_trained_model()
        result = get_analyzation_result(clf, vectorizer, raw_user_article_text)
        self.result_label.setText(
            f"Sentiment: {result['sentiment']}\nLabel: {result['label']}"
        )

