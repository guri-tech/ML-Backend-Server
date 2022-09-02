from prefect import Flow, Parameter
from tasks import (
    get_data,
    log_preprocessor,
    set_model,
    preprocessing,
    train_model,
    train_model_onepipeline,
    log_model,
    change_production_model,
)

with Flow("Hotel_train_onePipeline") as flow:
    eval_metric = Parameter("Evaluation Metric", "auc")
    model_name = "xgboost"
    df = get_data()
    model, params = set_model
    trained_model, metrics = train_model_onepipeline(model, df)
    current_version = log_model(trained_model, model_name, params, metrics, eval_metric)
    change_production_model(model_name, current_version, eval_metric)

with Flow("Hotel_train_preprocessor") as flow_pre:
    eval_metric = Parameter("Evaluation Metric", "auc")
    model_name = "xgboost"
    df = get_data()
    x, y, preprocessor = preprocessing(df)
    log_preprocessor(preprocessor)
    model, params = set_model
    trained_model, metrics = train_model(model, x, y)
    current_version = log_model(trained_model, model_name, params, metrics, eval_metric)
    change_production_model(model_name, current_version, eval_metric)

if __name__ == "__main__":
    # flow.register(project_name="hotel")
    # flow.run()
    flow_pre.run()
