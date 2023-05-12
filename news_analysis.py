import pickle
import pandas as pd

# from nltk import download as nltk_download
from nltk.corpus import stopwords
from nltk.corpus import words
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# nltk_download("words")
# nltk_download("stopwords")
# nltk_download("punkt")
# nltk_download("averaged_perceptron_tagger")
# nltk_download("wordnet")

english_words = set(words.words())
stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)


def get_part_of_speech_tag(token):
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV,
    }

    tag = pos_tag([token])[0][1][0].upper()
    return tag_dict.get(tag, wordnet.NOUN)


def get_tokens(text):
    return word_tokenize(text.lower())


def get_cleaned_tokens(text):
    tokens = get_tokens(text)
    return [
        token
        for token in tokens
        if (token in english_words and token not in stop_words)
    ]


def get_lemmatized_tokens(tokens):
    return [
        lemmatizer.lemmatize(token, get_part_of_speech_tag(token)) for token in tokens
    ]


def get_cleaned_text(text):
    cleaned_tokens = get_cleaned_tokens(text)
    lemmatized_tokens = get_lemmatized_tokens(cleaned_tokens)
    return " ".join(lemmatized_tokens)


try:
    f = open("my_classifier.pickle", "rb")
    vectorizer, clf = pickle.load(f)
    f.close()
except (OSError, IOError) as e:
    f = open("my_classifier.pickle", "wb")

    df = pd.read_csv("fake_or_real_news.csv")
    df["cleaned_text"] = df["text"].apply(get_cleaned_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned_text"], df["label"], test_size=0.2, random_state=42
    )

    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)

    clf = MultinomialNB()
    clf.fit(X_train_vect, y_train)

    pickle.dump((vectorizer, clf), f)
    f.close()

sentiment_analyser = SentimentIntensityAnalyzer()


def get_polarity_score(text):
    return sentiment_analyser.polarity_scores(text)["compound"]


def get_analyzation_result(article):
    cleaned_article = get_cleaned_text(article)
    polarity_score = get_polarity_score(cleaned_article)
    sentiment = "positive" if polarity_score > 0 else "negative"
    article_vect = vectorizer.transform([cleaned_article])
    prediction = clf.predict(article_vect)[0]
    return {"sentiment": sentiment, "label": prediction}


def get_text_file_content(file_path):
    with open(file_path, "r") as file:
        return file.read()
