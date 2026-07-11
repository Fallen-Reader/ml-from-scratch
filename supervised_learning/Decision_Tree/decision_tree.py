import numpy as np

class Node:
    def __init__(self):
        self.features = None
        self.threshold = None
        self.right = None
        self.left = None

        self.is_leaf = False
        self.value = None
    
    def __repr__(self):
        if self.is_leaf:
            return f"Leaf(predict)={self.value}"
        return f"split(features = {self.features}, threshold = {self.threshold})"

def gini_imp(y):
    m = len(y)
    if m==0:
        return 0.0
    
    imp = 1
    for c in np.unique(y):
        p = np.sum(y==c)/m
        imp -= p**2
    return imp

def weigthed_gini_imp(y_left,y_right):
    m = len(y_left)+len(y_right)

    w_left = len(y_left)/m
    w_right = len(y_right)/m

    return w_left*gini_imp(y_left)+w_right*gini_imp(y_right)

def best_split(x,y):
    m,n = x.shape

    best_feat = None
    best_gini = float('inf')
    best_thres = None

    for j in range(n):
        threshold =np.unique( x[:,j])
        for t in threshold:
            left_mask = x[:,j]<=t
            right_mask = ~left_mask

            y_left = y[left_mask]
            y_right = y[right_mask]

            if len(y_left)==0 or len(y_right)==0:
                continue

            g = weigthed_gini_imp(y_left,y_right)

            if g<best_gini:
                best_gini = g
                best_feat = j
                best_thres = t
        
        return best_feat,best_thres,best_gini
    
def build_tree(x,y,depth =0,maxDepth=5,min_sample=2):
    node = Node()

    if len(np.unique(y))==1:
        node.is_leaf = True
        node.value =y[0]
        return node
    
    if depth>=maxDepth:
        node.is_leaf = True
        node.value = np.bincount(y.astype(int)).argmax()
        return node
    
    if len(y)<min_sample:
        node.is_leaf = True
        node.value = np.bincount(y.astype(int)).argmax()
        return node
    
    feat,thres,gini = best_split(x,y)

    if feat is None:
        node.is_leaf = True
        node.value = np.bincount(y.astype(int)).argmax()
        return node
    
    node.features = feat
    node.threshold = thres

    l_mask = x[:,feat]<=thres
    r_mask = x[:,feat]>thres

    node.left = build_tree(x[l_mask],y[l_mask],depth+1)
    node.right = build_tree(x[r_mask],y[r_mask],depth+1)

    return node

def predict_one(node, x):
    
    if node.is_leaf:
        return node.value

    if x[node.features] <= node.threshold:
        return predict_one(node.left,  x)   #left
    else:
        return predict_one(node.right, x)   #right


def predict(root, X):
    return np.array([predict_one(root, x) for x in X])

def print_tree(node, feature_names=None, depth=0):
    indent = "  " * depth

    if node.is_leaf:
        print(f"{indent}Leaf -> predict class {node.value}")
        return

    fname = feature_names[node.features] if feature_names else f"feature_{node.features}"

    print(f"{indent}{fname} <= {node.threshold:.3f}?")
    print(f"{indent}  |-- YES (go left):")
    print_tree(node.left,  feature_names, depth+2)
    print(f"{indent}  └-- NO  (go right):")
    print_tree(node.right, feature_names, depth+2)

X_toy = np.array([
    [1.0, 2.0],    
    [1.5, 1.8],    
    [5.0, 8.0],    
    [6.0, 7.5],   
])
y_toy = np.array([0, 0, 1, 1])

print("="*50)
print("GINI CHECKS")
print("="*50)
print(f"Gini of pure node [0,0,0]:   {gini_imp(np.array([0,0,0])):.3f} ")
print(f"Gini of mixed node [0,0,1,1]:{gini_imp(np.array([0,0,1,1])):.3f} ")

print("\n" + "="*50)
print("BEST SPLIT ON TOY DATA")
print("="*50)
feat, thresh, g = best_split(X_toy, y_toy)
print(f"Best feature : {feat}")
print(f"Best threshold: {thresh}")
print(f"Weighted Gini : {g:.4f}")

print("\n" + "="*50)
print("TREE STRUCTURE")
print("="*50)
root = build_tree(X_toy, y_toy, maxDepth=3)
print_tree(root, feature_names=['feature_0', 'feature_1'])

print("\n" + "="*50)
print("PREDICTIONS")
print("="*50)
y_pred = predict(root, X_toy)
print(f"Predicted : {y_pred}")
print(f"Actual    : {y_toy}")
print(f"Accuracy  : {np.mean(y_pred == y_toy)*100:.0f}%")
