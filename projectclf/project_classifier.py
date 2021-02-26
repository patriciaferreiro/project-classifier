import numpy as np
import pandas as pd
from projectclf import utils


class CategoryClassifier:

    def __init__(self, categories: list, token_sizes: list):
        self.categories = categories
        self.token_sizes = token_sizes

    # check if any n-gram matches the categories
    def classify_category(self, input: list) -> None:
        # TODO: Improve efficiency when matching
        tokens = []
        for size in self.token_sizes:
            tokens.extend(utils.tokenize(input, n=size))
        # try to match token to env
        for token in tokens:
            if token in self.categories:
                return token
        return None


class ProjectCluster:

    def __init__(self, min_ngram: int, max_ngram: int, ignored_words: list) -> None:
        self.min_ngram = min_ngram
        self.max_ngram = max_ngram
        self.ignored_words = ignored_words

    def build_tdidf_matrix(self, words: list) -> np.matrix:
        from sklearn.feature_extraction.text import TfidfVectorizer

        # remove env substrings from corpus words
        corpus = [utils.replace_matches(word, self.ignored_words, '-') for word in words]
        # what if a word is shorter than max n-gram size?
        stop_words = ['-', '_']
        vectorizer = TfidfVectorizer(input=corpus, ngram_range=(self.min_ngram, self.max_ngram),
                                     lowercase=True, stop_words=set(stop_words),
                                     token_pattern=r'(?u)\b[A-Za-z]+\b')
        X = vectorizer.fit_transform(corpus)
        tfidf_matrix = X.todense()
        return tfidf_matrix

    # group similar projects together
    def group_projects(self, workspaces: list) -> list:
        from sklearn.cluster import AgglomerativeClustering

        # use whole corpus to create TF-IDF matrix
        tfidf_matrix = self.build_tdidf_matrix(workspaces)
        np.random.seed(0)  # seed to preserve clustering label order
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.005).fit(tfidf_matrix)
        groups = clustering.labels_
        return groups


class ProjectClassifier():

    def classify(self, input_col: str, irrelevant_words: list,
                 envs: list, env_sizes: list = [2, 3, 4],
                 min_ngram: int = 1, max_ngram: int = 4,
                 pdf: pd.DataFrame = None, path: str = None, ) -> pd.DataFrame:

        # token sizes must encompass all envs to match
        cc = CategoryClassifier(categories=envs, token_sizes=env_sizes)
        pc = ProjectCluster(min_ngram=min_ngram, max_ngram=max_ngram, ignored_words=irrelevant_words)
        # get data directly or read from path
        if path:
            pdf = utils.read_file_as_pdf(path)
        elif pdf is None:
            ValueError('You must input either a pandas DF or a path to data file.')
        # add prediction cols to pdf
        pdf['pred_env'] = [cc.classify_category(wk) for wk in pdf[input_col]]
        pdf['pred_group'] = pc.group_projects(pdf[input_col])
        return pdf


def main():
    # used for quick testing
    envs = ['pre', 'prod', 'qa', 'uat', 'test', 'tst', 'dev', 'dr']
    irrelevant_words = ['sourcing', 'analysis', 'processing', 'debug', 'analytics', 'ingest']
    # pass specific irrelevant words before envs, so they're matched first
    path = '../data/sample.csv'
    input_col = 'project_name'
    clf = ProjectClassifier()
    pdf = clf.classify(input_col=input_col, irrelevant_words=irrelevant_words + envs, envs=envs, path=path)
    pdf = pdf.sort_values(by=['pred_group'])
    return pdf


if __name__ == "__main__":
    main()
