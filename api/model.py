from joblib import load

def model():
    iris_clf = load("./iris.pkl")
    return iris_clf