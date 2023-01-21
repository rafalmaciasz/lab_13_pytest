from app import hello
from app import extract_sentiment
from app import text_contain_word
from app import mull_matrices
import pytest
import numpy as np


def test_hello():
    got = hello("Rafal")
    want = "Hello Rafal"

    assert got == want

testdata1 = ["I think today will be a great day"]

@pytest.mark.parametrize('sample', testdata1)
def test_extract_sentiment(sample):

    sentiment = extract_sentiment(sample)

    assert sentiment > 0

testdata2 = [
    ('There is a duck in this text', 'duck', True),
    ('There is nothing here', 'duck', False)
    ]

@pytest.mark.parametrize('sample, word, expected_output', testdata2)
def test_text_contain_word(sample, word, expected_output):

    assert text_contain_word(word, sample) == expected_output
    
testdata3 = [
    (np.array([[1, 1], [2, 3]]),
    np.array([[1, 2], [2, 3]]),
    np.array([[3, 5], [8, 13]]))
]

@pytest.mark.parametrize('matrix_1, matrix_2, result', testdata3)
def test_mull_matrices(matrix_1, matrix_2, result):
    
    assert (mull_matrices(matrix_1, matrix_2) == result).all