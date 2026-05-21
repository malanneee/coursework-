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
    df["sentence_count"] = df["text"].apply(lambda x: x.count(".") + 1)
    df["avg_sentence_length"] = df["word_count"] / df["sentence_count"]
    return df

data = pd.read_csv("balanced_cefr.csv")
data = add_features(data)

X_text = data["text"]
X_extra = data[[
    "text_length",
    "word_count",
    "avg_word_length",
    "sentence_count",
    "avg_sentence_length"
]]
y = data["level"]

X_text_train, X_text_test, X_extra_train, X_extra_test, y_train, y_test = train_test_split(
    X_text, X_extra, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

vectorizer = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 3),
    min_df=3,
    max_df=0.95,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_text_train)
X_test_tfidf = vectorizer.transform(X_text_test)

X_train_final = hstack([X_train_tfidf, X_extra_train.values])
X_test_final = hstack([X_test_tfidf, X_extra_test.values])
param_grid = {

    'n_estimators': [100, 150],

    'learning_rate': [0.05, 0.1],

    'max_depth': [3, 5],

}
grid = GridSearchCV(
    GradientBoostingClassifier(
      random_state=42
    ),
    param_grid,
    cv=3,
    scoring = 'f1_weighted',
    n_jobs = -1,
    verbose = 2
)

grid.fit(X_train_final, y_train)
print("Best parameters:")
print(grid.best_params_)
best_model = grid.best_estimator_

y_pred = best_model.predict(X_test_final)

print("Classification report:")
print(classification_report(y_test, y_pred))

def predict_level(text: str) -> str:
    text_df = pd.DataFrame({"text": [text]})
    text_df = add_features(text_df)

    text_tfidf = vectorizer.transform(text_df["text"])
    text_extra = text_df[[
        "text_length",
        "word_count",
        "avg_word_length",
        "sentence_count",
        "avg_sentence_length"
    ]].values

    text_final = hstack([text_tfidf, text_extra])

    return best_model.predict(text_final)[0]

example_text = "I like cooking simple meals at home. I often make pasta, soup, or salad. Cooking helps me save money and eat healthy food."

print("Predicted level:", predict_level(example_text))
