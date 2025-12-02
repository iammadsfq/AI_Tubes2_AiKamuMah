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
        # TODO: Implementasi rekursif pembentukan tree
        # 1. Cek stopping criteria (depth, purity, n_samples)
        # 2. Cari best split (Information Gain / Gain Ratio / Gini Index)
        # 3. Buat Node
        pass

    def _best_split(self, X, y, feat_idxs): # Gain
        # TODO: Logic untuk mencari split terbaik
        # Harus bisa handle Categorical DAN Numerical
        pass

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

    def _information_gain(self, y, left_indices, right_indices):
        parent_size = y.size
        left_y = y[left_indices]
        right_y = y[right_indices]

        left_proportion = left_y.size / parent_size
        right_proportion = right_y.size / parent_size

        child_entropy = left_proportion * self._entropy(left_y) + right_proportion * self._entropy(right_y)
        return self._entropy(y) - child_entropy

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
