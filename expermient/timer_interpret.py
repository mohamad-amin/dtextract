# %%
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from timy.settings import timy_config
timy_config.tracking = False

# %%
from dtextract.data.data import ID, NUM, CAT_RES

output = 'logs/newsgroups_3000_features.txt'  # The log file in which the log of running code will be written
path = '../data/20_newsgroups_3000_features.csv'  # Path to the input dataset in csv format
data_types = [ID] * 1 + [NUM] * 3000 + [CAT_RES]
has_header = True
isClassify = True

# %%
# decision tree extraction parameters
nComponents = 100  # Number of components (the gaussian mixtures)
maxSize = 32  # maximum tree size
nPts = 1000  # Number of points used in active sampling
nTestPts = 1000  # Number of test points used in Active sampling
maxDtSize = maxSize  # decision tree training parameters
tgtScore = None
minGain = 1e-3

# random forest parameters
nTrees = 100

# training data proportion
trainingProp = 0.7

# %%
from dtextract.util.log import *
from dtextract.data.data import readCsv

setCurOutput(output)
log('Parsing CSV...', INFO)
(df, res, resMap, catFeats, columnNames) = readCsv(path, has_header, data_types)
log('Done!', INFO)

# %%
from dtextract.data.data import split

log('Splitting into training and test...', INFO)
(trainDf, testDf) = split(df, trainingProp)
log('Done!', INFO)

# %%
from dtextract.data.data import constructDataMatrix

log('Constructing data matrices...', INFO)
(XTrain, yTrain, catFeatIndsTrain, numericFeatIndsTrain) = constructDataMatrix(trainDf, res, catFeats)
(XTest, yTest, catFeatIndsTest, numericFeatIndsTest) = constructDataMatrix(testDf, res, catFeats)
log('Done!', INFO)

# %%
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

log('Training random forest...', INFO)
rfConstructor = RandomForestClassifier if isClassify else RandomForestRegressor
rf = rfConstructor(n_estimators=nTrees)
rf.fit(XTrain, yTrain)
log('Done!', INFO)

# %%
from dtextract.util.util import f1Vec, mseVec

rfScoreFunc = f1Vec if isClassify else mseVec
rfTrainScore = rfScoreFunc(rf.predict, XTrain, yTrain)
rfTestScore = rfScoreFunc(rf.predict, XTest, yTest)
log('Training score: ' + str(rfTrainScore), INFO)
log('Test score: ' + str(rfTestScore), INFO)

# %%
from dtextract.core.learn import ParamsLearn
from dtextract.impl.simp import ParamsSimp
from dtextract.impl.funcs import getRfFunc

# Step 2: Set up decision tree extraction inputs
paramsLearn = ParamsLearn(tgtScore, minGain, maxSize)
paramsSimp = ParamsSimp(nPts, nTestPts, isClassify)

# Step 3: Function
rfFunc = getRfFunc(rf)

# %%
from dtextract.impl.dists import CategoricalGaussianMixtureDist as CGMD

dist = CGMD(XTrain, catFeatIndsTrain, numericFeatIndsTrain, nComponents)

# %%
from dtextract.impl.simp import learnDTSimp, genAxisAligned

# Step 5: Extract decision tree
dtExtract, dtMap = learnDTSimp(genAxisAligned, rfFunc, dist, paramsLearn, paramsSimp)

# %%
log('Decision tree:', INFO)
log(str(dtExtract), INFO)
log('Node count: ' + str(dtExtract.nNodes()), INFO)
log('DT in DOT language:', INFO)
log(str(dtExtract.toDotGraph(hasHeader=has_header, columnNames=columnNames)), INFO)

# %%
from dtextract.util.util import f1, mse

scoreFunc = f1 if isClassify else mse

dtExtractRelTrainScore = scoreFunc(dtExtract.eval, XTrain, rf.predict(XTrain))
dtExtractRelTestScore = scoreFunc(dtExtract.eval, XTest, rf.predict(XTest))

log('Relative training score: ' + str(dtExtractRelTrainScore), INFO)
log('Relative test score: ' + str(dtExtractRelTestScore), INFO)

dtExtractTrainScore = scoreFunc(dtExtract.eval, XTrain, yTrain)
dtExtractTestScore = scoreFunc(dtExtract.eval, XTest, yTest)

log('Training score: ' + str(dtExtractTrainScore), INFO)
log('Test score: ' + str(dtExtractTestScore), INFO)

# %%
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Step 6: Train a (greedy) decision tree
log('Training greedy decision tree', INFO)
maxLeaves = (maxDtSize + 1) // 2
dtConstructor = DecisionTreeClassifier if isClassify else DecisionTreeRegressor
dtTrain = dtConstructor(max_leaf_nodes=maxLeaves)
dtTrain.fit(XTrain, rfFunc(XTrain))
log('Done!', INFO)
log('Node count: ' + str(dtTrain.tree_.node_count), INFO)

# %%
dtTrainRelTrainScore = scoreFunc(lambda x: dtTrain.predict(x.reshape(1, -1)), XTrain, rf.predict(XTrain))
dtTrainRelTestScore = scoreFunc(lambda x: dtTrain.predict(x.reshape(1, -1)), XTest, rf.predict(XTest))

log('Relative training score: ' + str(dtTrainRelTrainScore), INFO)
log('Relative test score: ' + str(dtTrainRelTestScore), INFO)

# %%
dtTrainTrainScore = scoreFunc(lambda x: dtTrain.predict(x.reshape(1, -1)), XTrain, yTrain)
dtTrainTestScore = scoreFunc(lambda x: dtTrain.predict(x.reshape(1, -1)), XTest, yTest)

log('Training score: ' + str(dtTrainTrainScore), INFO)
log('Test score: ' + str(dtTrainTestScore), INFO)

# %%
from dtextract.interpreter.dt_interpreter import *

predictions, biases, contributions = interpret_tree(dtExtract, dtMap, XTest, yTest)
assert_interpretation(predictions, biases, contributions)

# %%
descriptions = ''
if has_header:
    # NOTE: here you should trim the headers array to be the array of names of input columns
    # with the same size as the number of columns. for example if the first row is and id row
    # and the last row is the label row, you can use:
    headers = list(pd.read_csv(path))[1:-1]
    headers = np.array(headers)
    exps = interpret_samples(rf, dtExtract, XTest, contributions, labels=headers)
    for i in range(len(exps)):
        descriptions += exps[i].get_description(5, True, headers)
else:
    exps = interpret_samples(rf, dtExtract, XTest, contributions)
    for i in range(len(exps)):
        descriptions += exps[i].get_description(5)

# %%
print(descriptions)

