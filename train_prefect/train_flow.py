from prefect import Flow, Parameter
from tasks import (
    get_data,
    preprocessing,
    set_model,
    train_model,
    log_model,
    change_production_model,
)

with Flow("Hotel_train_Model") as flow:
    eval_metric = Parameter("Evaluation Metric", "auc")

    df = get_data()
    x, y = preprocessing(df)
    model, params, model_name = set_model(1)
    trained_model, metrics = train_model(model, x, y)
    current_version = log_model(trained_model, model_name, params, metrics, eval_metric)
    change_production_model(model_name, current_version, eval_metric)

if __name__ == "__main__":
    flow.run()
