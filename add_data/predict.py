import os
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri(os.getenv("MLFLOW_SET_TRACKING_URI"))
mlflow.set_experiment("HOTEL-EXPERIMENT-NOPREPROCESSOR")


def predict(df):
    model_name = "RandomForestClassifier_onePipeline"
    # name_filter = f"name='{model_name}'"
    # client = MlflowClient()
    # model = client.search_model_versions(name_filter)

    # model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
    preprocessor = mlflow.sklearn.load_model(
        "s3://mlflow/5/52904b6648c64c08bd8c006231866d98/artifacts/preprocessor"
    )
    model = mlflow.sklearn.load_model(
        "s3://mlflow/5/925b1551c8e140aca40127b4f893c158/artifacts/RandomForestClassifier"
    )

    df_new = preprocessor.transform(df)
    pred, proba = model.predict(df_new), model.predict_proba(df_new)
    proba = [round(y, 2) for x in proba for y in x]  # 2 dim -> 1 dim

    # print('\n[preprocessor] :',dir(preprocessor))
    # print('\n[model] :', dir(model))
    print("predict", pred)
    print("proba", proba)
    print("feature_importances_", model.feature_importances_)

    return pred, proba[1]
