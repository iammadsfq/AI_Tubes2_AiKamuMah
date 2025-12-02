import numpy as np
import pickle

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature          # Index fitur yang dijadikan split
        self.threshold = threshold      # Nilai threshold (untuk numerik)
        self.left = left                # Child kiri
        self.right = right              # Child kanan
        self.value = value              # Class label (jika leaf node)

class DecisionTreeModel:
    def __init__(self, min_samples_split=2, max_depth=100, algorithm='id3'):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.algorithm = algorithm # 'id3', 'c4.5', atau 'cart'
        self.root = None

    def fit(self, X, y):
        """
        X: numpy array features
        y: numpy array target
        """
        # TODO: Handle Null Values sebelum building tree atau di dalam split
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples = X.shape[0]
        n_classes = np.unique(y).size

        if n_samples < self.min_samples_split or\
           n_classes <= 1 or\
           depth >= self.max_depth:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.array(range(X.shape[1]))
        best_feature, best_threshold, left_idxs, right_idxs = self._best_split(X, y, feat_idxs)

        if best_threshold is None:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        left_child = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right_child = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)

        return Node(feature=best_feature, threshold=best_threshold, left=left_child, right=right_child)

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        best_feature = 0
        best_threshold = None
        best_left_idxs = None
        best_right_idxs = None

        for feature in feat_idxs:
            column = X[:, feature]
            sorted_unique_column = np.sort(np.unique(column))
            for value in sorted_unique_column:
                left_idxs = np.where(column <= value)[0]
                right_idxs = np.where(column > value)[0]
                if left_idxs.size == 0 or right_idxs.size == 0:
                    continue
                gain = self._information_gain(y, left_idxs, right_idxs)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = value
                    best_left_idxs = left_idxs
                    best_right_idxs = right_idxs

        return (best_feature, best_threshold, best_left_idxs, best_right_idxs)


    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value

        # TODO: Logic traversal (cek threshold/kategori)
        pass

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    # Algorithm specific helper
    def _entropy(self, y):
        size = y.size
        if size == 0:
            return 0.0

        unique_value = np.unique_counts(y)
        entropy = 0
        for count in unique_value.counts:
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

    def _most_common_label(self, y):
        values, counts = np.unique(y, return_counts=True)
        return values[np.argmax(counts)]

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

    def _print_recursive(self, node, depth):
        # TODO: Logic Visualisasi Tree
        return

# def entropy(y):
#     unique_value = np.unique_counts(y)
#     entropy = 0
#     size = y.size
#     for count in unique_value.counts:
#         proportion = count / size
#         entropy += proportion * np.log2(proportion)

#     return entropy * -1

# def main():
#     y = np.array([0, 0, 0, 1])
#     print(entropy(y))

# if __name__ == "__main__":
#     main()
