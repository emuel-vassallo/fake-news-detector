# Fake News Detector App

## Description
A Python GUI app for detecting fake news and analysing sentiment. It allows users to select a text file for analysis and displays the results on the GUI.

## Preview
![](showcase.gif)

## Installation
To use this app, you will need to install the following python dependencies:
- nltk
- pandas
- PySide6
- scikit_learn
- vaderSentiment

They can be installed by executing the following command in your terminal:
```
pip install -r requirements.txt
```


## Usage
To use the app, follow these steps:
1. Launch the app by executing `python main.py` in your terminal.
2. Click the button with label "Select Article Text File" to select a text file to analyse.
3. Click the "Analyse" button to analyse the selected file.
4. The results of the analysis will be displayed in the GUI.

## Features
- Classification of input text as either "fake" or "real" news
- Analysis of the sentiment of the input text
- User-friendly GUI for easy interaction with the app

## Acknowledgments
- The dataset used for training the machine learning model can be found at: https://github.com/lutzhamel/fake-news.

