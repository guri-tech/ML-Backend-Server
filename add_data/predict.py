import os
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri(os.getenv("URI"))
mlflow.set_experiment("Default")


def predict(df):
    model_name = "xgboost"

    preprocessor = mlflow.sklearn.load_model(f"models:/preprocessor/Production")
    model = mlflow.sklearn.load_model(f"models:/{model_name}/Production")

    df_new = preprocessor.transform(df)
    pred, proba = model.predict(df_new), model.predict_proba(df_new)
    proba = [round(y, 2) for x in proba for y in x]  # 2 dim -> 1 dim

    # print('\n[preprocessor] :',dir(preprocessor))
    # print('\n[model] :', dir(model))
    print("predict", pred)
    print("proba", proba)
    print("feature_importances_", model.feature_importances_)

    return pred, proba[1]
