import sklearn.metrics as metrics
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

class Classifier:
    """Classifier model."""
    def __init__(self, model,k=3):
        if model == 'SVM':
            self.model = SVC(kernel='linear')
        elif model == 'RandomForest':
            self.model = RandomForestClassifier()
        elif model == 'KNN':
            self.model = KNeighborsClassifier(n_neighbors=k)
        elif model == 'LogisticRegression':
            self.model = LogisticRegression()
        else: 
            raise ValueError('Invalid model')
        
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)
    
    def metric(self, y_test, y_pred):
        accuracy = metrics.accuracy_score(y_test, y_pred)
        precision = metrics.precision_score(y_test, y_pred, average='weighted')
        recall = metrics.recall_score(y_test, y_pred, average='weighted')
        f1 = metrics.f1_score(y_test, y_pred, average='weighted')
        cm = metrics.confusion_matrix(y_test, y_pred, labels=self.model.classes_)
        return accuracy, precision, recall, f1, cm