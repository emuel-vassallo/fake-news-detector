import pandas as pd

# from nltk import download as nltk_download
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# nltk_download("stopwords")
# nltk_download("punkt")
# nltk_download("averaged_perceptron_tagger")
# nltk_download("wordnet")

df = pd.read_csv("fake_or_real_news.csv")


def get_cleaned_tokens(text):
    stop_words = stopwords.words("english")
    tokens = word_tokenize(text.lower())
    return [token for token in tokens if token not in stop_words]


def get_part_of_speech_tag(token):
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV,
    }

    tag = pos_tag([token])[0][1][0].upper()
    return tag_dict.get(tag, wordnet.NOUN)


lemmatizer = WordNetLemmatizer()


def get_lemmatized_tokens(tokens):
    return [
        lemmatizer.lemmatize(token, get_part_of_speech_tag(token)) for token in tokens
    ]


def get_cleaned_text(text):
    cleaned_tokens = get_cleaned_tokens(text)
    lemmatized_tokens = get_lemmatized_tokens(cleaned_tokens)
    return " ".join(lemmatized_tokens)


df["cleaned_text"] = df["text"].apply(get_cleaned_text)

X_train, X_test, y_train, y_test = train_test_split(
    df["cleaned_text"], df["label"], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

clf = MultinomialNB()
clf.fit(X_train_vect, y_train)


sentiment_analyser = SentimentIntensityAnalyzer()


def get_polarity_score(text):
    return sentiment_analyser.polarity_scores(text)["compound"]


def get_results(article):
    cleaned_article = get_cleaned_text(article)
    polarity_score = get_polarity_score(cleaned_article)
    sentiment = "positive" if polarity_score > 0 else "negative"
    article_vect = vectorizer.transform([cleaned_article])
    prediction = clf.predict(article_vect)[0]
    return {"sentiment": sentiment, "label": prediction}
