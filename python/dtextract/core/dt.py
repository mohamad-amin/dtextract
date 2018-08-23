# Copyright 2015-2016 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# A data structure representing an internal
# node in a decision tree. The branch field
# is the conditional of the tree, with True
# indicating left and False indicating right.
# In particular, branch must define a
# function eval : X -> bool.
#
# type params:
#  X : input domain
#  Y : output domain
#
# fields:
#  branch : {eval : X -> bool}
#  left : DTNode | {eval : X -> Y}
#  right : DTNode | {eval : X -> Y}
class DTNode:
    # initialization
    def __init__(self, branch, left, right, id):
        self.branch = branch
        self.left = left
        self.right = right
        self.id = id

    # evaluate the output func(x) for a given input x
    #
    # parameters/returns:
    #  x : X
    #  return : Y
    def eval(self, x):
        val = self.branch.eval(x)
        if not type(val) == bool:
            raise Exception('Invalid branch: ' + str(self.branch) + ', produced value: ' + str(val) + ', of type: ' + str(type(val)) + ', on input: ' + str(x))
        if val:
            return self.left.eval(x)
        else:
            return self.right.eval(x)

    def eval_leaf(self, x):
        val = self.branch.eval(x)
        if not type(val) == bool:
            raise Exception('Invalid branch: ' + str(self.branch) + ', produced value: ' + str(val) + ', of type: ' + str(type(val)) + ', on input: ' + str(x))
        if val:
            return self.left.eval_leaf(x)
        else:
            return self.right.eval_leaf(x)
        
    # number of child nodes
    #
    # parameters/returns:
    #  return : int (number of child nodes)
    def nNodes(self):
        return 1 + (self.left.nNodes() if self.left.__class__ == DTNode else 1) + (self.right.nNodes() if self.right.__class__ == DTNode else 1)

    def get_label(self):
        return str(self.branch)

    # String representation using Lisp-like syntax
    def __str__(self):
        return '((' + str(self.branch) + ') ' + str(self.left) + ' ' + str(self.right) + ')'

# A default data structure representing a leaf
# node in a decision tree. It defines a function
# eval : X -> Y mapping inputs reaching this
# leaf to outputs. It takes as input a function
# func : X -> Y and defines eval(x) = func(x).
#
# type params:
#  X : input domain
#  Y : output domain
#
# fields:
#  func : X -> Y
class DTLeaf:
    # initialization
    def __init__(self, func, id):
        self.func = func
        self.id = id
    
    # evaluate the output func(x) for a given input x
    #
    # parameters/returns:
    #  x : X
    #  return : Y
    def eval(self, x):
        return self.func(x)

    def eval_leaf(self, x):
        return self

    def get_label(self):
        return str(self)

# A default branch struct that takes as input a
# function func : X -> bool and defines eval(x) = func(x).
#
# type params:
#  X : input domain
#
# fields:
#  func : X -> bool
class DTBranch:
    def __init__(self, func):
        self.func = func
    
    # evaluate the branch func(x) for a given input x
    def eval(self, x):
        return self.func(x)

# A data structure representing a decision tree.
#
# type params:
#  X : input domain
#  Y : output domain
#
# fields:
#  root : DTNode | {eval : X -> Y}
class DT:
    # initialization
    def __init__(self, root):
        self.root = root
    
    # evaluate the output func(x) for a given input x
    #
    # parameters/returns:
    #  x : X
    #  return : Y
    def eval(self, x):
        return self.root.eval(x)

    def eval_leaf(self, x):
        return self.root.eval_leaf(x)

    # node count
    #
    # parameters/returns:
    #  return : int (number of nodes in the tree)
    def nNodes(self):
        return self.root.nNodes() if self.root.__class__ == DTNode else 1

    def toDotGraph(self):
        dotBook = 'digraph g {' + '\n'
        tabChar = '    '
        dotBook += tabChar + 'forcelabels=true;' + '\n'
        dotBook += DT._toDotLabel(str(0), self.root.get_label())
        dotBook += DT._toDotGraphInternal(self.root, 0)
        dotBook += '}'
        return dotBook

    @staticmethod
    def _toDotGraphInternal(node, num):
        if node.__class__ == DTNode:

            leftNodes = node.left.nNodes() if (node.left.__class__ == DTNode) else 1
            leftNodes += 1

            dotBook = DT._toDotLabel(str(num+1), node.left.get_label())
            dotBook += DT._toDotLabel(str(num+leftNodes), node.right.get_label())

            dotBook += DT._dotTo(str(num), str(num+1))
            dotBook += DT._dotTo(str(num), str(num+leftNodes))

            dotBook += DT._toDotGraphInternal(node.left, num+1)
            dotBook += DT._toDotGraphInternal(node.right, num+leftNodes)

            return dotBook

        else:
            return ''

    @staticmethod
    def _toDotLabel(name, label):
        tabChar = '    '
        return tabChar + name + ' [label="' + label + '"];' + '\n'

    @staticmethod
    def _dotTo(n_from, n_to):
        tabChar = '    '
        return tabChar + n_from + ' -> ' + n_to + ';' + '\n'

    # Convert to string.
    def __str__(self):
        return DT._strHelper(self.root, '')
    
    @staticmethod
    def _strHelper(node, prefix):
        tabChar = '    '
        if node.__class__ == DTNode:
            newPrefix = prefix + tabChar
            res = prefix + '(' + str(node.branch) + '\n'
            res += DT._strHelper(node.left, newPrefix) + '\n'
            res += DT._strHelper(node.right, newPrefix) + ')'
            return res
        else:
            return prefix + str(node)
