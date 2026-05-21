#версия финальная
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from scipy.sparse import hstack
def add_features(df):
    df["text_length"] = df["text"].apply(len)
    df["word_count"] = df["text"].apply(lambda x: len(x.split()))
    df["avg_word_length"] = df["text"].apply(
        lambda x: np.mean([len(w) for w in x.split()]) if x.split() else 0
    )

    return df

data = pd.read_csv("balanced_cefr.csv")
data = add_features(data)

X_text = data["text"]
X_extra = data[["text_length","word_count","avg_word_length"]]
y = data["level"]

X_text_train, X_text_test, X_extra_train, X_extra_test, y_train, y_test = train_test_split(
    X_text, X_extra, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

vectorizer = TfidfVectorizer(
    max_features=7000,
    ngram_range=(1, 3),
    min_df=3,
    max_df=0.95,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_text_train)
X_test_tfidf = vectorizer.transform(X_text_test)

X_train_final = hstack([X_train_tfidf, X_extra_train.values])
X_test_final = hstack([X_test_tfidf, X_extra_test.values])
classifier = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)

classifier.fit(X_train_final, y_train)
y_pred = classifier.predict(X_test_final)

print("Classification report:")
print(classification_report(y_test, y_pred))

def predict_level(text: str) -> str:
    text_df = pd.DataFrame({"text": [text]})
    text_df = add_features(text_df)

    text_tfidf = vectorizer.transform(text_df["text"])
    text_extra = text_df[["text_length", "word_count", "avg_word_length"]].values

    text_final = hstack([text_tfidf, text_extra])

    return classifier.predict(text_final)[0]

example_text = "In recent years, online learning has become very popular.Many courses are now available on the internet."

print("Predicted level:", predict_level(example_text))
