import numpy as np
from decision_tree import Node,build_tree,predict

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class GradientBoostedTrees:

    def __init__(self, n_trees=50, max_depth=3,
                 learning_rate=0.1, min_samples=2):
        self.n_trees       = n_trees
        self.max_depth     = max_depth
        self.learning_rate = learning_rate
        self.min_samples   = min_samples
        self.trees         = []
        self.base_pred     = None      

    def fit(self, X, y):
        
        p = np.mean(y)
        self.base_pred = np.log(p / (1 - p))  

        F = np.full(len(y), self.base_pred) 

        for i in range(self.n_trees):

           
            probs     = sigmoid(F)              
            residuals = y - probs              

            tree = build_tree(
                X, residuals,
                max_depth=self.max_depth,
                min_samples=self.min_samples
            )
            self.trees.append(tree)

           
            correction = predict(tree, X)       
            F += self.learning_rate * correction

            if (i+1) % 10 == 0:
                probs    = sigmoid(F)
                preds    = (probs >= 0.5).astype(int)
                acc      = np.mean(preds == y)
                print(f"Tree {i+1:>3}/{self.n_trees}  Train accuracy: {acc:.4f}")

    def predict_proba(self, X):
        
        F = np.full(X.shape[0], self.base_pred)

        
        for tree in self.trees:
            F += self.learning_rate * predict(tree, X)

        return sigmoid(F)    

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
    
