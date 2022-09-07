from prefect import Flow, Parameter
from prefect.run_configs import UniversalRun
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


with Flow(
    name="Hotel_train_onePipeline", run_config=UniversalRun(labels=["prod1"])
) as flow_pipeline:
    eval_metric = Parameter("Evaluation Metric", "auc")
    model_name = "xgboost_pipeline"
    df = get_data()
    model, params = set_model
    trained_model, metrics = train_model_onepipeline(model, df)
    current_version = log_model(trained_model, model_name, params, metrics, eval_metric)
    change_production_model(model_name, current_version, eval_metric)

with Flow(
    name="Hotel_train_preprocessor", run_config=UniversalRun(labels=["prod2"])
) as flow_with_preprocessor:
    eval_metric = Parameter("Evaluation Metric", "auc")
    model_name = "xgboost"
    # model_name = "baggingclf"

    df = get_data()
    x, y, preprocessor = preprocessing(df)
    log_preprocessor(preprocessor)
    model, params = set_model
    trained_model, metrics = train_model(model, x, y)
    current_version = log_model(trained_model, model_name, params, metrics, eval_metric)
    change_production_model(model_name, current_version, eval_metric)

if __name__ == "__main__":
    flow_pipeline.register(project_name="hotel")
    # flow_with_preprocessor.register(project_name="hotel")
    flow_with_preprocessor.run()
