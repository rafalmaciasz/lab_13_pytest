from textblob import TextBlob
import numpy as np

def hello(name):
    output = f'Hello {name}'
    return output


def extract_sentiment(text):
    text = TextBlob(text)

    return text.sentiment.polarity

def text_contain_word(word: str, text: str):
    return word in text

def mull_matrices(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    m, n = m1.shape
    if m == m2.shape[1] and n == m2.shape[0]:
        new_m = np.array([[np.sum(m1[i][:] * m2[:][j])\
            for j in range(n)] for i in range(m)])
    return new_m
        