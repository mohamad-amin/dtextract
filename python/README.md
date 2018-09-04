# Running the code throught run_jupyter.ipynb
You can use this jupyter notebook to easily run the code on your dataset.
First you must change the parameters mentioned below as your needs:
```python
output = TEST_OUTPUT  # The log file in which the log of running code will be written
path = TEST_PATH  # Path to the input dataset in csv format
data_types = TEST_DATA_TYPES
has_header = TEST_HAS_HEADER
nComponents = 100  # Number of components (the gaussian mixtures)
maxSize = 64  # maximum tree size
nPts = 2000  # Number of points used in active sampling
nTestPts = 2000  # Number of test points used in Active sampling
isClassify = True  # Weather the problem is a classification or regression problem
```
These parameters can be provided through the [consts_generated.py](https://github.com/mohamad-amin/dtextract/blob/master/python/dtextract/data/consts_generated.py) file.

**Note:** the input dataset must be shuffled if you want it to. This code's `split` function that splits the input dataset to trianing and testing dataset doesn't shuffle the rows before splitting them and does this splitting according to the indexes (i.e. if the dataset has 100 datapoints and the `trainingProp` is set to `0.7` the first 70 samples are chosen as training data and the last 30 samples will be counted as test data).

Then you can run the code line by line using ipython or something similar to train the blackbox model and approximate the decision tree and get the local explanations in the `descriptions` variable.
You also have access to the `biases`, `contributions` and `predictions` vectors that come from the tree interpreation algorithm ([described here](http://blog.datadive.net/random-forest-interpretation-with-scikit-learn/)).

If you want to add headers to the descriptions (local explanations), you must provide the `headers` array that is mentioned here (the last slot of the jupyter notebook):
```python
descriptions = ''
if has_header:
    # NOTE: here you should trim the headers array to be the array of names of input columns 
    # with the same size as the number of columns. for example if the first row is and id row 
    # and the last row is the label row, you can use: 
    # headers = list(pd.read_csv(path))[1:-1]
    headers = np.array(headers)
    exps = interpret_samples(rf, dtExtract, XTest, contributions, labels=headers)
    for i in range(len(exps)):
        descriptions += exps[i].get_description(5, True, headers)
else:
    exps = interpret_samples(rf, dtExtract, XTest, contributions)
    for i in range(len(exps)):
        descriptions += exps[i].get_description(5)
```
