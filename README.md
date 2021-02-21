# Project classifier
Takes a group of project names and:
 1. Classifies them into a given set of categories.
 2. Groups them by project.

## Project structure
```
.
├── README.md
├── docs
├── requirements.txt
├── src
│   └──main.py
└──tests
    └── test_main.py
```

## Implementation details
- Classification is done by attempting to match the project name n-grams to a predetermined category list.
If no match is found, the project is not classified.'

- Clustering by project is done by applying an Agglomerative clustering algorithm to the project name n-grams TD-IDF matrix.
