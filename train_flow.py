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


with Flow(name="Hotel_train_preprocessor") as h_flow:
    eval_metric = Parameter("Evaluation Metric", "auc")

    df = get_data()
    x, y, preprocessor, redis_vari_num = preprocessing(df)
    log_preprocessor(preprocessor)
    model, params, model_name = set_model
    trained_model, metrics = train_model(model, x, y)
    current_version = log_model(trained_model, model_name, params, metrics, eval_metric)
    change_production_model(
        model, model_name, current_version, eval_metric, redis_vari_num
    )

if __name__ == "__main__":

    h_flow.run()
