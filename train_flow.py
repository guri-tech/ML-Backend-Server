from prefect import Flow, Parameter
from prefect.run_configs import UniversalRun, LocalRun
from tasks import (
    get_data,
    log_preprocessor,
    preprocessing,
    train_model,
    log_model,
    change_production_model,
)


def flow():
    eval_metric = Parameter("Evaluation Metric", "auc")
    df = get_data()
    x, y, preprocessor, redis_vari_num = preprocessing(df)
    log_preprocessor(preprocessor)
    trained_model, metrics = train_model(model, x, y)
    current_version = log_model(trained_model, model_name, params, metrics, eval_metric)
    change_production_model(
        model, model_name, current_version, eval_metric, redis_vari_num
    )


with Flow(name="Hotel_DecisionTree", run_config=LocalRun(labels=["dt"])) as dt:

    from sklearn.tree import DecisionTreeClassifier

    params = {}
    model = DecisionTreeClassifier(**params)
    model_name = "DecisionTreeClassifier"

    flow()

with Flow(name="Hotel_RandomForest", run_config=LocalRun(labels=["rf"])) as rf:

    from sklearn.ensemble import RandomForestClassifier

    params = {"n_estimators": 100, "max_depth": 4, "min_samples_split": 3}
    model = RandomForestClassifier(**params)
    model_name = "RandomForestClassifier"

    flow()


with Flow(name="Hotel_svm", run_config=LocalRun(labels=["svm"])) as svm:

    from sklearn.svm import SVC

    params = {}
    model = SVC(probability=True, **params)
    model_name = "svm"

    flow()


if __name__ == "__main__":
    # dt.register(project_name="hotel")
    # rf.register(project_name="hotel")
    # svm.register(project_name="hotel")
    rf.run()
