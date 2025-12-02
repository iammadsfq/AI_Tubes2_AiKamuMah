import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import networkx as nx

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

        if x[node.feature] <= node.threshold:
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

    def _print_recursive(self, node, depth=0, prefix=""):
        if node.value is not None:
            print(str(node.value) + " (Leaf)")
        else:
            print("[Feature " + str(node.feature) + " <= " + str(node.threshold) + "]")
            new_prefix = prefix + "│   "
            print(prefix + "├── " + "Left: ", end="")
            self._print_recursive(node.left, depth + 1, new_prefix)
            new_prefix = prefix + "    "
            print(prefix + "└── " + "Right: ", end="")
            self._print_recursive(node.right, depth + 1, new_prefix)

    def _add_edges(self, G, node, parent=None, label=""):
        if node is None:
            return

        node_id = id(node)

        if node.value is not None:
            node_label = f"{node.value}\n(Leaf)"
            G.add_node(node_id, label=node_label, leaf=True)
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
            # Edge label in the middle
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
    # Simple test
    X = np.array([
    [2, 8, 60],
    [5, 7, 70],
    [6, 6, 70],
    [8, 5, 75],
    [3, 6, 55],
    [5, 7, 68],
    [7, 5, 80],
    [1, 9, 50],
    [9, 4, 85],
    [2, 7, 58],
])
    y = np.array(["Fail", "Fail", "Pass", "Pass", "Fail", "Pass", "Pass", "Fail", "Pass", "Fail"])

    tree = DecisionTreeModel(max_depth=5)
    tree.fit(X, y)
    predictions = tree.predict(np.array([[6, 6, 72]]))
    print("Predictions:", predictions)
    print("Actual:     ", ["Yes"])
    print("Accuracy:   ", np.mean(predictions == y))
    tree.print_tree()

if __name__ == "__main__":
    main()
