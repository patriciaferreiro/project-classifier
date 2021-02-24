import logging
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# TODO: Add function to generate / manage data
workspaces = ['playground', 'db-testing-01', 'asdpro', 'asddev', 'jordidev', 'qatest-env']
envs = ['pre', 'prod', 'qa', 'uat', 'test', 'tst', 'dev']


# TODO: Find a more efficient way!
def replace_matches(word: str, new_token: str = '') -> str:
    for env in envs:
        if env in word:
            return word.replace(env, new_token)
    return word


def tokenize(word: list, n: int = 3) -> list:
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
    def classify_category(self, input: list) -> None:
        # improve efficiency when matching
        tokens = tokenize(input, n=3)
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
        corpus = [replace_matches(word, '-') for word in words]

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
    import utils
    workspaces = ['playground', 'db-testing-01', 'asdpro', 'asddev', 'jordidev', 'qatest-env']
    envs = ['pre', 'pro', 'prod', 'qa', 'uat', 'test', 'tst', 'dev']

    cc = CategoryClassifier(categories=envs)
    pc = ProjectCluster()
    pdf = utils.read_as_pdf()
    pdf['pred_env'] = cc.classify_category(pdf['project_name'])
    pdf['pred_group'] = pc.group_projects(pdf['project_name'])


# TODO: Improve project structure
# https://towardsdatascience.com/building-package-for-machine-learning-project-in-python-3fc16f541693
def main():
    import utils

    envs = ['pre', 'pro', 'prod', 'qa', 'uat', 'test', 'tst', 'dev']
    cc = CategoryClassifier(categories=envs)
    pc = ProjectCluster()
    pdf = utils.read_as_pdf()
    pdf['pred_env'] = [cc.classify_category(wk) for wk in pdf['project_name']]
    pdf['pred_group'] = pc.group_projects(pdf['project_name'])
    print(pdf)
    return pdf


if __name__ == "__main__":
    main()
