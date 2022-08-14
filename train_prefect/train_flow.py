from prefect import Flow, Parameter
from tasks import (
    get_data,
    preprocessing,
    train_model,
    log_model,
    change_production_model,
)

from xgboost import XGBClassifier

params = {
    "booster": "gbtree",
    "objective": "binary:logistic",
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "n_jobs": -1,
}
model = XGBClassifier(use_label_encoder=False, **params)
model_name = model.__class__.__name__

with Flow("Model Training Flow") as flow:
    eval_metric = Parameter("Evaluation Metric", "auc")
    df = get_data()
    x, y, preprocessed_model = preprocessing(df)
    model, metrics = train_model(model, x, y)
    current_version = log_model(model, model_name, params, metrics, eval_metric)
    production_version = change_production_model(
        model_name, current_version, eval_metric
    )
flow.run()
