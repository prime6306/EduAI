from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import joblib

def train_models(X_train, y_train):
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    joblib.dump(nb, 'naive_bayes.pkl')

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    joblib.dump(lr, 'logistic_regression.pkl')

def predict(model_path, X_test):
    model = joblib.load(model_path)
    return model.predict(X_test)
