import logging
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# TODO: Find a more efficient way!
def replace_matches(word: str, matches: list, new_token: str = '') -> str:
    for m in matches:
        if m in word:
            # may be more than 1 match
            word = word.replace(m, new_token)
    return word


def tokenize(word: list, n: int = 3) -> list:
    from nltk import ngrams

    # cast to list... maybe there's a better way
    wk_split = list(ngrams(word, n))
    wk_ngram = [''.join(ngram) for ngram in wk_split]
    logging.info(f'The word "{word}" has been {n}-grammed! \nResult: {wk_ngram}')
    return wk_ngram


def read_file_as_pdf(path: str = 'data/sample.csv') -> pd.DataFrame:
    '''Reads a csv file from the data folder and returns it as a pandas data frame.'''
    data = pd.read_csv(path)
    logging.info(f'Pandas DF read from {path}.')
    return data
