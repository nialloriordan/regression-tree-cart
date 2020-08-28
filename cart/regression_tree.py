import numpy as np
from cart.utils import square_impurity


class TreeNode(object):
    """Tree Node class.
    
    Args:
        left: left node
        right: right node
        feature: feature used to split node
        cut: cut value used to split node
        prediction: node prediction
    """

    def __init__(self, prediction):
        self.left = None
        self.right = None
        self.feature = 0
        self.cut = 0
        self.prediction = prediction


class Regression_Tree(object):
    """Regression Tree class
    
    Args:
        max_depth: max tree depth    
    """

    def __init__(self, max_depth=np.inf):
        self.max_depth = max_depth
        self.depth = 0
        self.tree = None

    def fit(self, x_train, y_train):
        """ Build a regression tree
        
        Args:
            x_train: n x d matrix of data
            y_train: n-dimensional vector
        """
        self.tree = self._build_tree(x_train, y_train)

    def _build_tree(self, x_train, y_train):
        """Builds a regression tree recursively using the CART algorithm.

        Args:
            x_train: n x d matrix of data
            y_train: n-dimensional vector

        Returns:
            tree: root of decision tree
        """

        n, d = x_train.shape

        node = TreeNode(prediction=np.mean(y_train))

        if (
            (len(np.unique(y_train)) < 2)
            or (len(np.unique(x_train, axis=0)) < 2)
            or (self.depth == self.max_depth)
        ):
            return node

        feature, cut, bestloss = self._best_split_rule(x_train, y_train)

        idx_left = np.where(x_train[:, feature] <= cut)[0]
        idx_right = np.where(x_train[:, feature] > cut)[0]

        x_train_left, y_train_left = x_train[idx_left], y_train[idx_left]
        x_train_right, y_train_right = x_train[idx_right], y_train[idx_right]

        node.feature = feature
        node.cut = cut
        self.depth = self.depth + 1

        node.left = self._build_tree(x_train_left, y_train_left)
        node.right = self._build_tree(x_train_right, y_train_right)

        return node

    def _best_split_rule(self, x_train, y_train):
        """Find the best feature, cut value, and loss value.

        Args:
            x_train: n x d matrix of data points
            y_train: n-dimensional vector of labels

        Returns:
            feature: index of the best cut's feature
            cut: cut-value of the best cut
            bestloss: loss of the best cut
        """

        N, D = x_train.shape
        assert D > 0  # must have at least one dimension
        assert N > 1  # must have at least two samples

        bestloss = np.inf

        for i in range(D):
            x_i = x_train[:, i]
            cut_values = np.unique(x_i)
            x_i_sort_idx = cut_values.argsort()
            cut_values = cut_values[x_i_sort_idx]

            for i_cut in range(len(cut_values) - 1):
                cut_value = (cut_values[i_cut] + cut_values[i_cut + 1]) / 2

                y_idx_left = np.where(x_i < cut_value)[0]
                y_idx_right = np.where(x_i >= cut_value)[0]

                y_left = y_train[y_idx_left]
                y_right = y_train[y_idx_right]

                left_impurity = square_impurity(y_left)
                right_impurity = square_impurity(y_right)

                loss = left_impurity + right_impurity

                if loss < bestloss:
                    bestloss = loss
                    cut = cut_value
                    feature = i

        return feature, cut, bestloss

    def predict(self, x):
        """Evaluates x using decision tree root.

        Args:
            x: n x d matrix of data points

        Returns:
            pred: n-dimensional vector of predictions
        """
        pred = np.ones(len(x))
        for i, x_i in enumerate(x):
            pred[i] = None
            node = self.tree
            while (node.left is not None) or (node.right is not None):
                if x_i[node.feature] <= node.cut:
                    node = node.left
                else:
                    node = node.right
            pred[i] = node.prediction

        return pred
