from prefect import Flow, Parameter
from prefect.run_configs import UniversalRun
from tasks import (
    get_data,
    log_preprocessor,
    set_model,
    preprocessing,
    train_model,
    log_model,
    change_production_model,
)


with Flow(name="Hotel_train_preprocessor") as flow:
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
    Flow.register(project_name="hotel")
    flow.run()
