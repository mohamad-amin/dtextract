from ..impl.base import *
from ..impl.simp import *


def interpret_tree(dt, dtMap, XTrain, yTrain, XTest, yTest):

    # log('Leaves:', INFO)
    mLeaves = []
    for i in range(len(XTest)):
        mLeaves.append(dt.eval_leaf(XTest[i]).id)
    # log(str(mLeaves) + '\n', INFO)

    # log('Paths:', INFO)
    paths = _get_tree_paths(dt, dtMap, 0)
    for path in paths:
        path.reverse()
    # log(str(paths), INFO)

    # log('Leaf to path:', INFO)
    leaf_to_path = {}
    # map leaves to paths
    for path in paths:
        leaf_to_path[path[-1]] = path
    # log(str(leaf_to_path) + '\n', INFO)

    values = _get_values(dt, XTrain, yTrain)
    normalizer = {k: sum(v) for k, v in values.items()}
    for node_id in values:
        for i in range(len(XTrain[0])):
            values[node_id][i] = values[node_id][i] / float(normalizer[node_id])
    # log('Normalized values:', INFO)
    # log(str(values) + '\n', INFO)

    biases = np.tile(values[paths[0][0]], (XTest.shape[0], 1))
    log('Biases:', INFO)
    log(str(biases), INFO)
    line_shape = (XTest.shape[1], len(values[0])) # Note: this is hacky :D
    # log('Line shape:', INFO)
    # log(str(line_shape) + '\n', INFO)

    feature_index = {}
    for node_id in dtMap:
        node = dtMap[node_id]
        if not _is_leaf(node):
            feature_index[node_id] = node.branch.ind
    contributions = []

    unique_leaves = np.unique(mLeaves)
    unique_contributions = {}

    nvalues = {}
    for node_id in values:
        nvalues[node_id] = np.array(values[node_id])
    # log('Converted values:')
    # log(str(values))

    # log('Feature index:')
    # log(str(feature_index))

    # log('Unique leaves:')
    # log(str(unique_leaves) + '\n')

    for row, leaf in enumerate(unique_leaves):
        path = leaf_to_path[leaf]
        contribs = np.zeros(line_shape)
        for i in range(len(path) - 1):
            value_from_parent = nvalues[path[i+1]]
            value_from_child = nvalues[path[i]]
            contrib = value_from_parent - value_from_child
            feature_row = feature_index[path[i]]
            contribs[feature_row] += contrib
        unique_contributions[leaf] = contribs

    for row, leaf in enumerate(mLeaves):
        contributions.append(unique_contributions[leaf])

    log('Contibs:', INFO)
    log(str(contributions) + '\n', INFO)

    log('Sum over axis 1:')
    log(str(np.sum(contributions, axis=1)))

    log('Prediction:')
    log(str(biases + np.sum(contributions, axis=1)))


def _get_values(dt, XTrain, yTrain):
    values = {}
    for i in range(len(XTrain)):
        node = dt.root
        while True:
            if node.id in values:
                values[node.id][_class(yTrain[i])] += 1
            else:
                values[node.id] = [0] * len(XTrain[0])
                values[node.id][_class(yTrain[i])] += 1
            if not _is_leaf(node):
                val = node.branch.eval(XTrain[i])
                if not type(val) == bool:
                    raise Exception('Invalid branch :(')
                if val:
                    node = node.left
                else:
                    node = node.right
            else:
                break
    return values


def _class(y):
    return int(round(y))


def _get_tree_paths(dt, dtMap, node_id, depth=0):
    """
    Returns all paths through the tree as list of node_ids
    """
    if _is_leaf(dtMap[node_id]):
        raise ValueError("Invalid node_id %s" % str(dtMap[node_id]))

    left_child_id = node_id*2+1
    right_child_id = node_id*2+2

    # paths = None
    # internal = True

    if _is_leaf(dtMap[left_child_id]):
        paths = [[left_child_id, node_id]]
    else:
        paths = _get_tree_paths(dt, dtMap, left_child_id, depth=depth + 1)
        for path in paths:
            path.append(node_id)

    if _is_leaf(dtMap[right_child_id]):
        paths += [[right_child_id, node_id]]
    else:
        right_paths = _get_tree_paths(dt, dtMap, right_child_id, depth=depth + 1)
        for path in right_paths:
            path.append(node_id)
        paths += right_paths

    # if not internal:
    #     if paths is None:
    #         paths = [[node_id]]
    #     else:
    #         paths += [[node_id]]

    return paths


def _is_leaf(node):
    if node is None:
        print('WTF')
    return node.__class__ == LeafNode
