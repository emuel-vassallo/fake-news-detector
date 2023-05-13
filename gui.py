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
    QGridLayout,
)
from news_analysis import (
    is_model_trained,
    get_trained_model,
    get_text_file_content,
    get_analyzation_result,
    save_result_in_file,
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
        self.title_label.setFont(QFont("Arial", 22, QFont.Bold))
        self.box_layout.addWidget(self.title_label, alignment=Qt.AlignHCenter)

        self.box_layout.addSpacing(26)

        self.is_model_trained_label = QLabel(self)
        self.is_model_trained_label.setText(
            'Model trained and ready. Select an article and click "Analyse" to start.'
            if is_model_trained()
            else 'Model training and analysis will begin after clicking "Analyse". Please note that it may take some time.'
        )
        self.is_model_trained_label.setStyleSheet("color: #808080")
        self.is_model_trained_label.setFont(QFont("Arial", 12))
        self.box_layout.addWidget(
            self.is_model_trained_label, alignment=Qt.AlignHCenter
        )

        self.box_layout.addSpacing(26)

        self.file_layout = QHBoxLayout()
        self.box_layout.addLayout(self.file_layout)

        self.selected_file_label = QLabel(self)
        self.selected_file_label.setMaximumWidth(160)
        self.selected_file_label.setText("Select Article Text File:")
        self.selected_file_label.setFont(QFont("Arial", 11))

        self.file_dialog_button = QPushButton("No file selected", self)
        self.file_dialog_button.setMaximumWidth(380)
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

        self.submit_button = QPushButton("Analyse", self)
        self.submit_button.setFixedSize(160, 40)
        self.submit_button.setFont(QFont("Arial", 11, QFont.Bold))
        self.submit_button.setStyleSheet(
            """
            QPushButton {
                background-color: #007bff;
                color: #fff;
                border-color: #0062cc;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #0069d9;
            }
            QPushButton:pressed {
                background-color: #0062cc;
            }
        """
        )
        self.submit_button.clicked.connect(self.analyse_file)
        self.submit_button.setCursor(Qt.PointingHandCursor)
        self.box_layout.addWidget(self.submit_button, alignment=Qt.AlignHCenter)

        self.box_layout.addSpacing(26)

        self.sentiment_label = QLabel(self)
        self.sentiment_label.setText("Sentiment")
        self.sentiment_label.setFont(QFont("Arial", 11, QFont.Bold))

        self.sentiment_result_label = QLabel(self)
        self.sentiment_result_label.setFont(QFont("Arial", 11))

        self.fake_news_label = QLabel(self)
        self.fake_news_label.setText("Real or Fake")
        self.fake_news_label.setFont(QFont("Arial", 11, QFont.Bold))

        self.result_label = QLabel(self)
        self.result_label.setAlignment(Qt.AlignLeft)
        self.result_label.setFont(QFont("Arial", 12))

        self.fake_news_result_label = QLabel(self)
        self.fake_news_result_label.setFont(QFont("Arial", 11))

        self.result_layout = QGridLayout()

        self.box_layout.addSpacing(26)

        self.analysis_result_label = QLabel(self)
        self.analysis_result_label.setText("Analysation Results")
        self.analysis_result_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.analysis_result_label.setAlignment(Qt.AlignCenter)
        self.result_layout.addWidget(self.analysis_result_label, 0, 0, 1, 2)

        self.result_layout.addWidget(
            self.sentiment_label, 1, 0, alignment=Qt.AlignCenter
        )
        self.result_layout.addWidget(
            self.sentiment_result_label, 2, 0, alignment=Qt.AlignCenter
        )
        self.result_layout.addWidget(
            self.fake_news_label, 1, 1, alignment=Qt.AlignCenter
        )
        self.result_layout.addWidget(
            self.fake_news_result_label, 2, 1, alignment=Qt.AlignCenter
        )

        self.box_layout.addLayout(self.result_layout)

        self.box_layout.setAlignment(Qt.AlignCenter)

        self.setCentralWidget(self.widget)

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

    def analyse_file(self):
        raw_user_article_text = get_text_file_content(self.selected_file_path)
        vectorizer, clf = get_trained_model()
        result = get_analyzation_result(clf, vectorizer, raw_user_article_text)
        sentiment = result["sentiment"]
        label = result["label"]
        accuracy = result["confidence"]

        save_result_in_file(raw_user_article_text, sentiment, label, accuracy)

        if sentiment == "negative":
            sentiment_emoji = "😞"
            sentiment_bg_color = "#392029"
            sentiment_text_color = "#EB9CA6"
            sentiment_border_color = "#EB9CA6"
        elif sentiment == "positive":
            sentiment_emoji = "😊"
            sentiment_bg_color = "#183C36"
            sentiment_text_color = "#47FFBC"
            sentiment_border_color = "#47FFBC"
        else:
            sentiment_emoji = "😐"
            sentiment_bg_color = "#343826"
            sentiment_text_color = "#E5E566"
            sentiment_border_color = "#E5E566"

        self.sentiment_result_label.setText(
            f"{sentiment_emoji} {sentiment.capitalize()}"
        )
        self.sentiment_result_label.setStyleSheet(
            f"""
            background-color: {sentiment_bg_color};       
            color: {sentiment_text_color}; 
            border: 1px solid {sentiment_border_color}; 
            padding: 6px;
            border-radius: 8px;
        """
        )

        if label == "FAKE":
            fake_news_emoji = "🚫"
            fake_news_bg_color = "#392029"
            fake_news_text_color = "#EB9CA6"
            fake_news_border_color = "#EB9CA6"
        else:
            fake_news_emoji = "✅"
            fake_news_bg_color = "#183C36"
            fake_news_text_color = "#47FFBC"
            fake_news_border_color = "#47FFBC"

        if fake_news_bg_color:
            self.fake_news_result_label.setStyleSheet(
                f"""
                background-color: {fake_news_bg_color};       
                color: {fake_news_text_color}; 
                border: 1px solid {fake_news_border_color}; 
                padding: 6px;
                border-radius: 8px;
            """
            )
            
        self.fake_news_result_label.setText(
            f"{fake_news_emoji} The article is {accuracy}% {label.lower()}"
        )
