import numpy as np
from decision_tree import Node,weigthed_gini_imp,predict


def bootstrap_sample(X, y):
    
    m    = len(X)
    idxs = np.random.choice(m, size=m, replace=True) 
    return X[idxs], y[idxs]


def build_tree_random(X, y, depth=0, max_depth=5,
                      min_samples=2, max_features=None):
    
    node = Node()

    if len(np.unique(y)) == 1:
        node.is_leaf = True
        node.value   = y[0]
        return node

    if depth >= max_depth or len(y) < min_samples:
        node.is_leaf = True
        node.value   = np.bincount(y.astype(int)).argmax()
        return node

    n = X.shape[1]

    
    if max_features is None:
        max_features = int(np.sqrt(n))     

    feat_idxs = np.random.choice(n, size=max_features, replace=False)
    
    best_g      = float('inf')
    best_feat   = None
    best_thresh = None

    for j in feat_idxs:                  
        thresholds = np.unique(X[:, j])

        for t in thresholds:
            left_mask  = X[:, j] <= t
            right_mask = ~left_mask
            y_left     = y[left_mask]
            y_right    = y[right_mask]

            if len(y_left) == 0 or len(y_right) == 0:
                continue

            g = weigthed_gini_imp(y_left, y_right)
            if g < best_g:
                best_g      = g
                best_feat   = j
                best_thresh = t

    if best_feat is None:
        node.is_leaf = True
        node.value   = np.bincount(y.astype(int)).argmax()
        return node

    node.features   = best_feat
    node.threshold = best_thresh

    left_mask  = X[:, best_feat] <= best_thresh
    right_mask = ~left_mask

    node.left  = build_tree_random(X[left_mask],  y[left_mask],
                                   depth+1, max_depth, min_samples, max_features)
    node.right = build_tree_random(X[right_mask], y[right_mask],
                                   depth+1, max_depth, min_samples, max_features)
    return node


class RandomForest:

    def __init__(self, n_trees=100, max_depth=5,
                 min_samples=2, max_features=None):
        self.n_trees     = n_trees
        self.max_depth   = max_depth
        self.min_samples = min_samples
        self.max_features = max_features
        self.trees       = []              

    def fit(self, X, y):
        self.trees = []

        for i in range(self.n_trees):
            
            X_boot, y_boot = bootstrap_sample(X, y)

            tree = build_tree_random(
                X_boot, y_boot,
                max_depth=self.max_depth,
                min_samples=self.min_samples,
                max_features=self.max_features
            )
            self.trees.append(tree)

            if (i+1) % 10 == 0:
                print(f"Built tree {i+1}/{self.n_trees}")

    def predict_(self, X):
        
        tree_preds = np.array([predict(tree, X) for tree in self.trees])

        final_preds = []
        for i in range(X.shape[0]):
            votes      = tree_preds[:, i]
            prediction = np.bincount(votes.astype(int)).argmax()
            final_preds.append(prediction)

        return np.array(final_preds)

    def feature_importance(self, n_features):
        
        counts = np.zeros(n_features)

        def count_splits(node):
            if node is None or node.is_leaf:
                return
            counts[node.feature] += 1
            count_splits(node.left)
            count_splits(node.right)

        for tree in self.trees:
            count_splits(tree)

        return counts / counts.sum()