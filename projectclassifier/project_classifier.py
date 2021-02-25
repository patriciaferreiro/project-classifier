import numpy as np
import pandas as pd
from projectclassifier import utils

ENVS = ['pre', 'pro', 'prod', 'qa', 'uat', 'test', 'tst', 'dev']


class CategoryClassifier:

    def __init__(self, categories=ENVS):
        self.categories = categories

    # check if any n-gram matches the categories
    def classify_category(self, input: list) -> None:
        # improve efficiency when matching
        tokens = utils.tokenize(input, n=3)
        for token in tokens:
            if token in self.categories:
                return token

        return None


class ProjectCluster:

    def __init__(self) -> None:
        pass

    def build_tdidf_matrix(self, words: list) -> np.matrix:
        from sklearn.feature_extraction.text import TfidfVectorizer

        # remove env substrings from corpus words
        corpus = [utils.replace_matches(word, '-') for word in words]
        # what if a word is shorter than max n-gram size?
        stop_words = ['-', '_']
        vectorizer = TfidfVectorizer(input=corpus, ngram_range=(1, 2),
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

    def classify(self, input_col: str='project_name', pdf: pd.DataFrame = None, path: str = None) -> pd.DataFrame:

        cc = CategoryClassifier()
        pc = ProjectCluster()
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
    clf = ProjectClassifier()
    pdf = clf.classify(path='../data/sample.csv')
    return pdf


if __name__ == "__main__":
    main()
