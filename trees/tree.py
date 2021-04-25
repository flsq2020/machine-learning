
from abc import ABC, abstractmethod
import numpy as np


class TreeNode:
    def __init__(self,
                 split_info,
                 left=None,
                 right=None,
                 children=[],
                 type='binary'):

        self.feature = split_info[0]
        self.threshold = split_info[1]

        assert isinstance(type, str), 'parameter `type` must be str'

        if type.lower() == 'binary':
            self.left = left
            self.right = right
        else:
            self.children = children


class TreeLeaf:
    def __init__(self, value):
        self.value = value


class DecisionTree(ABC):
    def __init__(self,
                 max_depth,
                 criterion,
                 classifier,
                 seed=None,
                 precision=None):
        self.depth = 0
        self.max_depth = max_depth
        self.criterion = criterion
        self.classifier = classifier
        self.seed = seed
        self.precision = precision

    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError

    @abstractmethod
    def importance_score(self):
        raise NotImplementedError


class ID3Tree(DecisionTree):
    def __init__(self,
                 max_depth=None,
                 precision=None,
                 seed=None
                 ):
        super().__init__(max_depth, 'infoGain', True, seed, precision)
        self.root = None

    def fit(self, X, Y):
        assert isinstance(X, np.ndarray), \
            'the type of input sample format error, must be `numpy.ndarray`'
        n, m = X.shape
        self.n_classes = max(Y) + 1
        self.max_depth = self.max_depth if self.max_depth else m
        self.root = self._grow(X, Y)

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def predict_class_probs(self, X):
        return np.array([self._traverse(x, self.root, True) for x in X])

    def _grow(self, X, Y, cur_depth=0):
        if len(set(Y)) == 1:
            prob = np.zeros(self.n_classes)
            prob[Y[0]] = 1.0
            return TreeLeaf(prob)

        if cur_depth >= self.max_depth:
            p = np.bincount(Y, minlength=self.n_classes) / len(Y)
            return TreeLeaf(p)

        cur_depth += 1
        self.depth = max(self.depth, cur_depth)

        feat_id, thresh = self._segment(X, Y)

        sample_idx = [np.argwhere(X[:, feat_id] == t).flatten() for t in thresh]
        children = [self._grow(X[idx, :], Y[idx], cur_depth) for idx in sample_idx]

        return TreeNode((feat_id, thresh), None, None, children, type='multi')

    def _segment(self, X, Y):
        gains = [self._impurity_gain(Y, np.unique(X[:, i]), X[:, i]) for i in range(X.shape[-1])]
        split_idx = gains.index(max(gains))
        return split_idx, np.unique(X[:, split_idx])

    def _impurity_gain(self, Y, thresholds, feat_vals):
        n = len(Y)
        parent_loss = entropy(Y)
        children = [np.argwhere(feat_vals==t).flatten() for t in thresholds]
        length = list(map(len, children))
        children_entropy = list(map(lambda x: entropy(Y[x]), children))
        child_loss = sum(list(map(lambda x: (x[0] / n) * x[1], zip(length, children_entropy))))
        ig = parent_loss - child_loss
        return ig

    def _traverse(self, X, node, prob=False):
        if isinstance(node, TreeLeaf):
            return node.value if prob else node.value.argmax()
        next_node = node.children[np.argwhere(node.threshold==X[node.feature]).flatten()[0]]
        return self._traverse(X, next_node, prob)

    def importance_score(self):
        pass


class CART(DecisionTree):
    def __init__(self,
                 max_depth=None,
                 classifier=True,
                 n_feats=None,
                 seed=None,
                 precision=None):
        super().__init__(max_depth, 'gini', classifier, seed, precision)
        if seed:
            np.random.seed(seed)
        self.root = None
        self.n_feats = n_feats
        self.depth = 0

    def fit(self, X, Y):
        n, m = X.shape
        self.max_depth = self.max_depth if self.max_depth else np.inf
        self.n_classes = max(Y) + 1 if self.classifier else None
        self.n_feats = m if not self.n_feats else min(self.n_feats, m)
        self.root = self._grow(X, Y)

    def predict(self, X):
        return np.array([self._traverse(x, self.root) for x in X])

    def predict_class_probs(self, X):
        assert self.classifier, "`predict_class_probs` undefined for classifier = False"
        return np.array([self._traverse(x, self.root, True) for x in X])

    def _grow(self, X, Y, cur_depth=0):
        if len(set(Y)) == 1:
            if self.classifier:
                prob = np.zeros(self.n_classes)
                prob[Y] = 1.0
            return TreeLeaf(prob) if self.classifier else TreeLeaf(Y[0])

        if cur_depth >= self.max_depth:
            v = np.mean(Y, axis=0)
            if self.classifier:
                v = np.bincount(Y, minlength=self.n_classes) / len(Y)
            return TreeLeaf(v)

        cur_depth += 1
        self.depth = max(self.depth, cur_depth)

        n, m = X.shape
        feat_idxs = np.random.choice(m, self.n_feats, replace=False)

        feat, threshold = self._segment(X, Y, feat_idxs)

        l_idx = np.argwhere(X[:, feat] <= threshold).flatten()
        r_idx = np.argwhere(X[:, feat] > threshold).flatten()
        left = self._grow(X[l_idx, :], Y[l_idx], cur_depth)
        right = self._grow(X[r_idx, :], Y[r_idx], cur_depth)

        return TreeNode((feat, threshold), left, right)

    def _segment(self, X, Y, feat_idxs):
        best_gain = -np.inf
        split_idx, split_threshold = None, None
        for i in feat_idxs:
            feat_vals = X[:, i]
            vals = np.unique(feat_vals)
            thresholds = (vals[:-1] + vals[1:]) / 2 if len(vals) > 1 else vals
            gains = np.array([self._impurity_gain(Y, feat_vals, t) for t in thresholds])

            if gains.max() > best_gain:
                best_gain = gains.max()
                split_idx = i
                split_threshold = thresholds[gains.argmax()]
        return split_idx, split_threshold

    def _impurity_gain(self, Y, feat_vals, t):
        loss = gini if self.classifier else mse
        parent_loss = loss(Y)

        l_idx = np.argwhere(feat_vals <= t).flatten()
        r_idx = np.argwhere(feat_vals > t).flatten()

        l = Y[l_idx]
        r = Y[r_idx]
        n_l, r_l, n = len(l), len(r), len(Y)

        if n_l == 0 or r_l == 0:
            return 0

        child_loss = loss(l) * (n_l / n) + loss(r) * (r_l / n)
        return parent_loss - child_loss

    def _traverse(self, X, node, prob=False):
        if isinstance(node, TreeLeaf):
            if self.classifier:
                return node.value if prob else node.value.argmax()
            return node.value
        if X[node.feature] <= node.threshold:
            return self._traverse(X, node.left, prob)
        return self._traverse(X, node.right, prob)

    def importance_score(self):
        pass


def mse(y):
    return np.mean((y - np.mean(y)) ** 2)


def entropy(y):
    hist = np.bincount(y)
    ps = hist / np.sum(hist)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])


def gini(y):
    hist = np.bincount(y)
    N = np.sum(hist)
    return 1 - sum([(i / N) ** 2 for i in hist])


if __name__ == '__main__':
    id3tree = ID3Tree()
    cart = CART()
    X = np.array([[0, 0, 0, 0],
                   [0, 0, 0, 1],
                   [0, 1, 0, 1],
                   [0, 1, 1, 0],
                   [0, 0, 0, 0],
                   [1, 0, 0, 0],
                   [1, 0, 0, 1],
                   [1, 1, 1, 1],
                   [1, 0, 1, 2],
                   [1, 0, 1, 2],
                   [2, 0, 1, 2],
                   [2, 0, 1, 1],
                   [2, 1, 0, 1],
                   [2, 1, 0, 2],
                   [2, 0, 0, 0]])
    Y = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0])
    id3tree.fit(X, Y)
    cart.fit(X, Y)
    print(id3tree.predict(X))
    # print(id3tree.predict_class_probs(X))
    print(cart.predict(X))
    print(cart.root.left.left)




