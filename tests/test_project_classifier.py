from src.project_classifier import tokenize
from src.project_classifier import CategoryClassifier
from src.project_classifier import ProjectCluster
from src.utils import *
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def test_tokenize():
    '''Creates an n-gram of a word and returns it as a list.'''
    word = 'jordi-test'
    ngram = 3
    res = tokenize(word, ngram)
    expectation = ['jor', 'ord', 'rdi', 'di-', 'i-t', '-te', 'tes', 'est']
    assert res == expectation


def test_classify_category():
    '''Classifies workspaces by trying to match their n-gram to an env.'''
    envs = ['pre', 'pro', 'prod', 'qa', 'test', 'tst', 'dev']
    workspaces = ['playground', 'db-testing-01', 'asdpro', 'asddev', 'jordidev', 'qatest-env']
    res = []
    expectation = [None, 'test', 'pro', 'dev', 'dev', 'test']
    cc = CategoryClassifier(categories=envs)
    for wk in workspaces:
        wk_tokenized = tokenize(wk, n=3)
        wk_tokenized.extend(tokenize(wk, n=4))
        pred = cc.classify_category(wk_tokenized)
        res.append(pred)
    assert res == expectation


def test_group_projects():
    '''Groups given workspaces by name similarity.'''
    workspaces = ['ga-playground', 'db-testing-01', 'asdpro', 'asddev', 'jordidev', 'qa-test']
    expectation = [4, 3, 0, 0, 1, 2]
    pc = ProjectCluster()
    # group_projects returns a numpy.ndarray, therefore casting to list
    res = (pc.group_projects(workspaces)).tolist()
    assert res == expectation


def test_predict_pdf():
    '''Predicts env and project group from a pandas dataframe.'''
    pdf = read_as_pdf()
