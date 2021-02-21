import logging
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

workspaces = ['playground', 'db-testing-01', 'asdpro', 'asddev', 'jordidev', 'qatest-env']
envs = ['pre', 'pro', 'prod', 'qa', 'test', 'tst', 'dev']


def remove_matches(word: str) -> str:
    for env in envs:
        if env in word:
            return word.replace(env, '')
    return word


def tokenize(word: list, n: int) -> list:
    from nltk import ngrams
    # turn workspace into n-grams
    # cast to list... maybe there's a better way
    wk_split = list(ngrams(word, n))
    wk_ngram = [''.join(ngram) for ngram in wk_split]
    logging.info(f'The word "{word}" has been {n}-grammed! \nResult: {wk_ngram}')
    return wk_ngram


class CategoryClassifier:

    def __init__(self, categories):
        self.categories = categories

    # check if any n-gram matches the categories
    def classify_category(self, tokens: list) -> None:
        # improve efficiency when matching
        for token in tokens:
            if token in self.categories:
                return token
        return None


class ProjectCluster:

    def __init__(self) -> None:
        pass

    def build_tdidf_matrix(self, words: list) -> np.matrix:
        from sklearn.feature_extraction.text import TfidfVectorizer
        stop_words = ['-', '_']
        # remove env substrings from corpus words
        corpus = [remove_matches(word) for word in words]
        # what if a word is shorter than max n-gram size?
        vectorizer = TfidfVectorizer(input=corpus, ngram_range=(1, 5),
                                     lowercase=True, stop_words=set(stop_words))
        X = vectorizer.fit_transform(corpus)
        tfidf_matrix = X.todense()
        return tfidf_matrix

    # group similar projects together
    def group_projects(self, workspaces: str) -> list:
        from sklearn.cluster import AgglomerativeClustering
        # use whole corpus to create TF-IDF matrix
        tfidf_matrix = self.build_tdidf_matrix(workspaces)
        np.random.seed(0)  # seed to preserve clustering label order
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.005).fit(tfidf_matrix)
        groups = clustering.labels_
        return groups


def __main__():
    categories, num_groups = envs, 3
    cc = CategoryClassifier(categories)
    pc = ProjectCluster(num_groups)
    env_preds = [cc.classify_category(wk) for wk in workspaces]
    project_preds = pc.group_projects
    return [(env, project) for env, project in zip(env_preds, project_preds)]
