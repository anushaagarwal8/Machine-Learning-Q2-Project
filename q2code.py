import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin


def load_data(filename, n_features=100000):
    data = []
    with open(filename) as f:
        for line in f:
            row = np.zeros(n_features)
            if line.strip():
                features = line.strip().split()
                for feature in features:
                    if ':' in feature:
                        idx, val = feature.split(':')
                        row[int(idx) - 1] = float(val) 
                    else:
                        row[int(feature) - 1] = 1.0
            data.append(row)
    return np.array(data)

data = load_data('dorothea_train.data')
labels = np.loadtxt('dorothea_train.labels')
X_val = load_data('dorothea_valid.data')
y_val = np.loadtxt('dorothea_valid.labels')

X_train, X_test, y_train, y_test = train_test_split( # splitting training data into train/test
    data, labels, test_size=0.2, random_state=42, stratify=labels
)

class EntropyWeightedRandomForest(BaseEstimator, ClassifierMixin):
    
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.classes = None
        self.n_classes = None
        self.rf = None
        
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)  
        self.rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        self.rf.fit(X, y)
        return self
    
    def calc_weights(self, X):
        n_samples = X.shape[0]
        n_trees = self.n_estimators
        weights = np.zeros((n_samples, n_trees))

        for i, tree in enumerate(self.rf.estimators_): # for each tree
            probs = tree.predict_proba(X) # proportions
            for j in range(n_samples):
                p_safe = np.clip(probs[j], 1e-10, 1.0) # avoid log errors
                H = -np.sum(p_safe * np.log2(p_safe)) # entropy
                weights[j, i] = np.exp(-10 * H)

        return weights
    
    def predict_proba(self, X):
        n_samples = X.shape[0]
        weights = self.calc_weights(X)
        weighted_probs = np.zeros((n_samples, self.n_classes))

        for i, tree in enumerate(self.rf.estimators_):
            tree_probs = tree.predict_proba(X)
            for j in range(n_samples):
                weighted_probs[j] += weights[j, i] * tree_probs[j]  # weight probability

        weight_sums = weights.sum(axis=1, keepdims=True)
        weighted_probs = weighted_probs / weight_sums
        return weighted_probs
    
    def predict(self, X):
        proba = self.predict_proba(X) # make prediction based on final weighted probabilities
        return self.classes[np.argmax(proba, axis=1)]

n_estimators_list = [50, 100]
max_depth_list = [5, 10, 15, 20, None]

for n_est in n_estimators_list:
    for max_d in max_depth_list:
        print("\n")
        print(f"Testing: n_estimators={n_est}, max_depth={max_d}")
        
        rf_std = RandomForestClassifier(
            n_estimators=n_est,
            max_depth=max_d,
            random_state=42,
        )
        rf_std.fit(X_train, y_train)
        
        y_train_pred_std = rf_std.predict(X_train)
        y_val_pred_std = rf_std.predict(X_val)
        y_test_pred_std = rf_std.predict(X_test)

        train_acc_std = accuracy_score(y_train, y_train_pred_std)
        val_acc_std = accuracy_score(y_val, y_val_pred_std)
        test_acc_std = accuracy_score(y_test, y_test_pred_std)
        train_prec_std = precision_score(y_train, y_train_pred_std)
        val_prec_std = precision_score(y_val, y_val_pred_std)
        test_prec_std = precision_score(y_test, y_test_pred_std)
        train_rec_std = recall_score(y_train, y_train_pred_std)
        val_rec_std = recall_score(y_val, y_val_pred_std)
        test_rec_std = recall_score(y_test, y_test_pred_std)
        trainf1_std = f1_score(y_train, y_train_pred_std)
        valf1_std = f1_score(y_val, y_val_pred_std)
        testf1_std = f1_score(y_test, y_test_pred_std)
        
        print("\nStandard RF:")
        print(f"  Training Accuracy:   {train_acc_std:.4f}")
        print(f"  Validation Accuracy: {val_acc_std:.4f}")
        print(f"  Test Accuracy:       {test_acc_std:.4f}")
        print(f"  Test Precision:      {test_prec_std:.4f}")
        print(f"  Test Recall:         {test_rec_std:.4f}")
        print(f"  Test F1-Score:       {testf1_std:.4f}")
        print(f"  Test Confusion Matrix:\n {confusion_matrix(y_test, y_test_pred_std)}")
        
        rf_ent = EntropyWeightedRandomForest(
            n_estimators=n_est,
            max_depth=max_d,
            random_state=42
        )
        rf_ent.fit(X_train, y_train)
        
        y_train_pred_ent = rf_ent.predict(X_train)
        y_val_pred_ent = rf_ent.predict(X_val)
        y_test_pred_ent = rf_ent.predict(X_test)
        
        train_acc_ent = accuracy_score(y_train, y_train_pred_ent)
        val_acc_ent = accuracy_score(y_val, y_val_pred_ent)
        test_acc_ent = accuracy_score(y_test, y_test_pred_ent)
        train_prec_ent = precision_score(y_train, y_train_pred_ent)
        val_prec_ent = precision_score(y_val, y_val_pred_ent)
        test_prec_ent = precision_score(y_test, y_test_pred_ent)
        train_rec_ent = recall_score(y_train, y_train_pred_ent)
        val_rec_ent = recall_score(y_val, y_val_pred_ent)
        test_rec_ent = recall_score(y_test, y_test_pred_ent)
        trainf1_ent = f1_score(y_train, y_train_pred_ent)
        valf1_ent = f1_score(y_val, y_val_pred_ent)
        testf1_ent = f1_score(y_test, y_test_pred_ent)
        
        print("\nWeighted RF:")
        print(f"  Training Accuracy:   {train_acc_ent:.4f}")
        print(f"  Validation Accuracy: {val_acc_ent:.4f}")
        print(f"  Test Accuracy:       {test_acc_ent:.4f}")
        print(f"  Test Precision:      {test_prec_ent:.4f}")
        print(f"  Test Recall:         {test_rec_ent:.4f}")
        print(f"  Test F1-Score:       {testf1_ent:.4f}")
        print(f"  Test Confusion Matrix:\n {confusion_matrix(y_test, y_test_pred_ent)}")

        test_improvement_acc = test_acc_ent - test_acc_std
        test_improvement_f1 = testf1_ent - testf1_std
        
        print(f"  Test Accuracy Improvement:       {test_improvement_acc:.4f}")
        print(f"  Test F1-Score Improvement:       {test_improvement_f1:.4f}")