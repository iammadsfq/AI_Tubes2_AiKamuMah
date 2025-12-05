import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import networkx as nx
from sklearn.base import BaseEstimator, ClassifierMixin

class Node:
    def __init__(self, is_categorical=False, feature=None, threshold=None, left=None, left_branch_majority=True, right=None, value=None):
        self.is_categorical = is_categorical # True jika membandingkan kategori
        self.feature = feature          # Index fitur yang dijadikan split
        self.threshold = threshold      # Nilai threshold (untuk numerik)
        self.left = left                # Child kiri
        self.left_branch_majority = left_branch_majority # True jika branch kiri memiliki lebih banyak sample
        self.right = right              # Child kanan
        self.value = value              # Class label (jika leaf node)

class DecisionTreeModel(BaseEstimator, ClassifierMixin):
    def __init__(self, min_samples_split=2, max_depth=100, algorithm='id3', ccp_alpha=0.0):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.algorithm = algorithm # 'id3', 'c4.5', atau 'cart'
        self.ccp_alpha = ccp_alpha
        self.root = None
        self.n_features = None

    def fit(self, X, y):
        """
        X: numpy array features
        y: numpy array target
        """
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        if hasattr(y, "to_numpy"):
            y = y.to_numpy()

        if len(y.shape) > 1:
            y = y.ravel()

        if X.shape[0] == 0 or y.size == 0:
            raise ValueError("Cannot fit model with empty dataset. X and y must have at least one sample.")

        self.n_features = X.shape[1]
        self.root = self._grow_tree(X, y)

        if self.ccp_alpha > 0:
            self._ccp_prune(X, y)

        return self

    # Missing Value Handler
    def _is_missing(self, value):
        return pd.isna(value)

    def _get_missing_mask(self, column):
        if np.issubdtype(column.dtype, np.number):
            return np.isnan(column)
        return pd.isna(column)

    # Categorical helper
    def _is_categorical_feature(self, column):
        return not np.issubdtype(column.dtype, np.number)

    def _grow_tree(self, X, y, depth=0):
        n_samples = X.shape[0]
        n_classes = np.unique(y).size

        depth_limit_reached = (self.max_depth is not None and depth >= self.max_depth)

        if n_samples < self.min_samples_split or \
           n_classes <= 1 or \
           depth_limit_reached:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        feat_idxs = np.array(range(X.shape[1]))
        best_feature, best_threshold, left_idxs, right_idxs, is_categorical = self._best_split(X, y, feat_idxs)

        if best_threshold is None or left_idxs is None or right_idxs is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        left_idxs = np.asarray(left_idxs)
        right_idxs = np.asarray(right_idxs)

        left_child = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right_child = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)

        left_branch_majority = left_idxs.size >= right_idxs.size
        return Node(is_categorical=is_categorical, feature=best_feature, threshold=best_threshold, left=left_child, left_branch_majority=left_branch_majority, right=right_child)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        best_feature = 0
        best_threshold = None
        best_left_idxs = None
        best_right_idxs = None
        best_is_categorical = False

        for feature in feat_idxs:
            column = X[:, feature]
            is_categorical = self._is_categorical_feature(column)

            missing_mask = self._get_missing_mask(column)
            known_idxs = np.where(~missing_mask)[0]
            missing_idxs = np.where(missing_mask)[0]

            if known_idxs.size == 0:
                continue

            known_column = column[known_idxs]
            sorted_unique_column = np.sort(np.unique(known_column))

            for value in sorted_unique_column:
                if is_categorical:
                    left_known_mask = known_column == value
                    right_known_mask = known_column != value
                else:
                    left_known_mask = known_column <= value
                    right_known_mask = known_column > value

                left_idxs = known_idxs[left_known_mask]
                right_idxs = known_idxs[right_known_mask]

                if left_idxs.size == 0 or right_idxs.size == 0:
                    continue

                gain = 0
                if self.algorithm == 'id3':
                    gain = self._information_gain(y, left_idxs, right_idxs)
                elif self.algorithm == 'c4.5':
                    gain = self._gain_ratio(y, left_idxs, right_idxs)
                elif self.algorithm == 'cart':
                    gain = self._gini_gain(y, left_idxs, right_idxs)

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = value
                    best_is_categorical = is_categorical

                    if missing_idxs.size > 0:
                        if left_idxs.size >= right_idxs.size:
                            best_left_idxs = np.concatenate([left_idxs, missing_idxs])
                            best_right_idxs = right_idxs
                        else:
                            best_left_idxs = left_idxs
                            best_right_idxs = np.concatenate([right_idxs, missing_idxs])
                    else:
                        best_left_idxs = left_idxs
                        best_right_idxs = right_idxs

        return (best_feature, best_threshold, best_left_idxs, best_right_idxs, best_is_categorical)


    def predict(self, X):
        if self.root is None:
            raise ValueError("Model has not been fitted yet. Call fit() before predict().")

        if hasattr(X, "to_numpy"):
            X = X.to_numpy()

        if X.shape[0] == 0:
            return np.array([])

        if X.shape[1] != self.n_features:
            raise ValueError(f"Feature dimension mismatch. Model was trained with {self.n_features} features, but input has {X.shape[1]} features.")

        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value

        feature_value = x[node.feature]

        if self._is_missing(feature_value):
            if node.left_branch_majority:
                return self._traverse_tree(x, node.left)
            else:
                return self._traverse_tree(x, node.right)

        if node.is_categorical:
            if feature_value == node.threshold:
                return self._traverse_tree(x, node.left)
            else:
                return self._traverse_tree(x, node.right)
        else:
            if feature_value <= node.threshold:
                return self._traverse_tree(x, node.left)
            else:
                return self._traverse_tree(x, node.right)

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    # Algorithm specific helper
    def _entropy(self, y):
        size = y.size
        if size == 0:
            return 0.0

        _, counts = np.unique(y, return_counts=True)
        entropy = 0
        for count in counts:
            proportion = count / size
            entropy += proportion * np.log2(proportion)

        return entropy * -1

    def _information_gain(self, y, left_idxs, right_idxs):
        parent_size = y.size
        left_y = y[left_idxs]
        right_y = y[right_idxs]

        left_proportion = left_y.size / parent_size
        right_proportion = right_y.size / parent_size

        child_entropy = left_proportion * self._entropy(left_y) + right_proportion * self._entropy(right_y)
        return self._entropy(y) - child_entropy

    def _split_information(self, left_idxs, right_idxs):
        left_size = left_idxs.size
        right_size = right_idxs.size

        if left_size == 0 or right_size == 0:
            return 0.0

        total_size = left_size + right_size
        left_proportion = left_size / total_size
        right_proportion = right_size / total_size

        return -((left_proportion) * np.log2(left_proportion) + (right_proportion) * np.log2(right_proportion))

    def _gain_ratio(self, y, left_idxs, right_idxs):
        info_gain = self._information_gain(y, left_idxs, right_idxs)
        split_info = self._split_information(left_idxs, right_idxs)

        if split_info == 0:
            return 0

        return info_gain / split_info

    def _gini(self, y):
        size = y.size
        if size == 0:
            return 0.0

        _, counts = np.unique(y, return_counts=True)
        gini = 0
        for count in counts:
            proportion = count / size
            gini += proportion * proportion

        return 1 - gini

    def _gini_gain(self, y, left_idxs, right_idxs):
        parent_size = y.size
        left_y = y[left_idxs]
        right_y = y[right_idxs]

        left_proportion = left_y.size / parent_size
        right_proportion = right_y.size / parent_size

        child_gini = left_proportion * self._gini(left_y) + right_proportion * self._gini(right_y)
        return self._gini(y) - child_gini


    def _most_common_label(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]


    # Pruning Optimization
    def _count_leaves(self, node):
        if node.value is not None:
            return 1

        return self._count_leaves(node.left) + self._count_leaves(node.right)


    def _subtree_error(self, node, X, y):
        y_pred = np.array([self._traverse_tree(x, node) for x in X])
        wrong = 0
        for i in range(y.size):
            if y[i] != y_pred[i]:
                wrong += 1

        return wrong / y.size


    def _leaf_error(self, y):
        common_class = self._most_common_label(y)
        wrong = 0
        for val in y:
            if val != common_class:
                wrong += 1

        return wrong / y.size


    def _effective_alpha(self, node, X, y):
        if node.value is not None:
            return float('inf')

        leaf_err = self._leaf_error(y)
        subtree_err = self._subtree_error(node, X, y)
        n_leaves = self._count_leaves(node)

        if n_leaves <= 1:
            return float('inf')

        return (leaf_err - subtree_err) / (n_leaves - 1)


    def _find_weakest_link(self, node, X, y):
        if node.value is not None:
            return (float('inf'), None, None, None)

        current_alpha = self._effective_alpha(node, X, y)
        min_alpha = current_alpha
        weakest_node = node
        weakest_X = X
        weakest_y = y

        column = X[:, node.feature]
        if node.is_categorical:
            left_mask = column == node.threshold
        else:
            left_mask = column <= node.threshold

        missing_mask = self._get_missing_mask(column)
        if node.left_branch_majority:
            left_mask = left_mask | missing_mask

        right_mask = ~left_mask

        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]

        left_alpha, left_node, left_X, left_y = self._find_weakest_link(node.left, X_left, y_left)
        if left_alpha < min_alpha:
            min_alpha = left_alpha
            weakest_node = left_node
            weakest_X = left_X
            weakest_y = left_y

        right_alpha, right_node, right_X, right_y = self._find_weakest_link(node.right, X_right, y_right)
        if right_alpha < min_alpha:
            min_alpha = right_alpha
            weakest_node = right_node
            weakest_X = right_X
            weakest_y = right_y

        return (min_alpha, weakest_node, weakest_X, weakest_y)

    def _prune_node(self, node, y):
        node.value = self._most_common_label(y)
        node.left = None
        node.right = None
        node.feature = None
        node.threshold = None

    def _ccp_prune(self, X, y):
        while True:
            min_alpha, weakest_node, weakest_X, weakest_y = self._find_weakest_link(self.root, X, y)

            if weakest_node is None or \
               min_alpha > self.ccp_alpha:
                break

            if weakest_node is self.root:
                self._prune_node(self.root, y)
                break

            self._prune_node(weakest_node, weakest_y)

    @staticmethod
    def load_model(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    # Bonus Visualisasi
    def print_tree(self):
        if not self.root:
            print("Tree belum dilatih!")
            return
        self._print_recursive(self.root, depth=0)

    def _print_recursive(self, node, depth=0, prefix=""):
        if node.value is not None:
            print(str(node.value) + " (Leaf)")
        else:
            if node.is_categorical:
                print("[Feature " + str(node.feature) + " == " + str(node.threshold) + "]")
            else:
                print("[Feature " + str(node.feature) + " <= " + str(node.threshold) + "]")
            new_prefix = prefix + "│   "
            print(prefix + "├── " + "Yes: ", end="")
            self._print_recursive(node.left, depth + 1, new_prefix)
            new_prefix = prefix + "    "
            print(prefix + "└── " + "No: ", end="")
            self._print_recursive(node.right, depth + 1, new_prefix)

    def _add_edges(self, G, node, parent=None, label=""):
        if node is None:
            return

        node_id = id(node)

        if node.value is not None:
            node_label = f"{node.value}\n(Leaf)"
            G.add_node(node_id, label=node_label, leaf=True)
        else:
            if node.is_categorical:
                node_label = f"{node.feature} == {node.threshold}"
            else:
                node_label = f"{node.feature} <= {node.threshold}"
            G.add_node(node_id, label=node_label, leaf=False)

        if parent is not None:
            G.add_edge(parent, node_id, label=label)

        if node.left:
            self._add_edges(G, node.left, node_id, "Yes")
        if node.right:
            self._add_edges(G, node.right, node_id, "No")

    def _calculate_pos(self, G, root, leftmost, width, vert_gap, vert_loc, pos):
        pos[root] = (leftmost + width / 2, vert_loc)
        children = list(G.successors(root))
        if len(children) != 0:
            dx = width / len(children)
            nextx = leftmost
            for child in children:
                pos = self._calculate_pos(G, child, nextx, dx, vert_gap, vert_loc - vert_gap, pos)
                nextx += dx
        return pos

    def draw_tree(self):
        if self.root is None:
            return

        G = nx.DiGraph()
        self._add_edges(G, self.root)

        labels = nx.get_node_attributes(G, 'label')
        edge_labels = nx.get_edge_attributes(G, 'label')
        leaf_nodes = [n for n, attr in G.nodes(data=True) if attr['leaf']]
        internal_nodes = [n for n in G.nodes() if n not in leaf_nodes]

        pos = self._calculate_pos(G, id(self.root), 0, 1., 0.2, 0, {})

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_axis_off()

        for (u, v, d) in G.edges(data=True):
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            ax.plot([x1, x2], [y1, y2], 'k-', zorder=1)
            xm, ym = (x1 + x2) / 2, ((y1 + y2) / 2) + 0.03
            ax.text(xm, ym, d['label'], color='black', fontsize=9, ha='center', va='center')

        for n in G.nodes():
            x, y = pos[n]
            node_label = labels[n]
            leaf = n in leaf_nodes
            bbox_props = dict(boxstyle="round,pad=0.3", fc="lightgreen" if leaf else "lightblue", ec="black", lw=1)
            ax.text(x, y, node_label, ha='center', va='center', fontsize=10, bbox=bbox_props, zorder=2)

        plt.tight_layout()
        plt.show()

def main():
    pass

if __name__ == "__main__":
    main()
