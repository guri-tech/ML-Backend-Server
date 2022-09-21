from prefect import Flow, Parameter
from prefect.run_configs import UniversalRun, LocalRun
from tasks import (
    get_data,
    log_preprocessor,
    set_model,
    preprocessing,
    train_model,
    log_model,
    change_production_model,
)


with Flow(name="Hotel_DecisionTree", run_config=LocalRun(labels=["dt"])) as dt:

    from sklearn.tree import DecisionTreeClassifier

    params = {}
    model = DecisionTreeClassifier(**params)
    model_name = "DecisionTreeClassifier"
    eval_metric = Parameter("Evaluation Metric", "auc")

    df = get_data()
    x, y, preprocessor, redis_vari_num = preprocessing(df)
    log_preprocessor(preprocessor)
    # model, params, model_name = set_model
    trained_model, metrics = train_model(model, x, y)
    current_version = log_model(trained_model, model_name, params, metrics, eval_metric)
    change_production_model(
        model, model_name, current_version, eval_metric, redis_vari_num
    )


with Flow(name="Hotel_RandomForest", run_config=LocalRun(labels=["rf"])) as rf:

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
    dt.register(project_name="hotel")
    rf.register(project_name="hotel")
