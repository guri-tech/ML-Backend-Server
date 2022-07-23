import os
import joblib

def saveModel(model, filename):
    BASE_DIR = os.path.dirname(__file__)
    file_loc = os.path.join(f'{BASE_DIR}/../trained_model/', filename)
    joblib.dump(model, file_loc)