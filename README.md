Decision Tree Extraction (DTExtract)
=====

DTExtract is a tool for extracting model explanations in the form of decision trees. More precisely, given

- blackbox access to a model (i.e., for a given input, produce the corresponding output),
- a sampling distribution over the input space,

then DTExtract constructs a decision tree approximating that model.

### IMPORTANT NOTE
The work on instant local interpretations of blackbox models is under *dev-tree-interpreter* branch, not in *master* branch. Please refer there, to see the code and experiments on **instant local interpretations of blackbox models**. Samples of the work that are available in the experiments section in that branch:
![Contributions](https://s3.amazonaws.com/media-p.slid.es/uploads/992266/images/6748999/Screen_Shot_2019-11-07_at_1.14.53_AM.png)
![Prediction Decomposition](https://s3.amazonaws.com/media-p.slid.es/uploads/992266/images/6749004/Screen_Shot_2019-11-07_at_1.15.07_AM.png)


Table of Contents
=====
0. Prerequisites
1. Setting Up DTExtract
2. Using DTExtract

Prerequisites
=====

DTExtract has been tested using Python **3.7**. DTExtract depends on numpy, scipy, scikit-learn, and pandas.

Setting Up DTExtract
=====

Run `setup.sh` to set up the datasets used in the examples that come with DTExtract.

Using DTExtract
=====

See `python/dtextract/examples/iris.py` for an example using a dataset from the UCI machine learning repository with the goal of classifying Iris flowers. The dataset is located at `data/iris.zip` ([download link](https://archive.ics.uci.edu/ml/datasets/Iris)). To run this example, run

    $ cd python
    $ python -m dtextract.examples.iris

Similarly, see `python/dtextract/examples/diabetes.py` for an example using a diabetes readmissions dataset. The dataset is located at `data/dataset_diabetes.zip` ([download link](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008)). To run this example, run

    $ cd python
    $ python -m dtextract.examples.diabetes

Finally, see `python/dtextract/examples/wine.py` for an example using a dataset from the UCI machine learning repository with the goal of classifying wines. The dataset is located at `data/wine.zip` ([download link](https://archive.ics.uci.edu/ml/datasets/Wine)). To run this example, run

    $ cd python
    $ python -m dtextract.examples.wine
