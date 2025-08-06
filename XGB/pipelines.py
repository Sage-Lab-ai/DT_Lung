from .TabularForecasting import *
from .Dataset import *
from .BaselineModels import *
from .utils import *
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
import shutil


def forecasting_pipeline(
        training_dataset: Dataset,
        savefolder: str,
        temporal_order: TemporalOrder | None = None,
        xgb_hyperparameters_to_search: dict | None = None,
        rf_hyperparameters_to_search: dict | None = None,
        grid_search_settings: dict | None = None):

    temporal_order = temporal_order or TemporalOrder([])
    xgb_hyperparameters_to_search = xgb_hyperparameters_to_search or default_xgb_hyperparameters_to_search
    rf_hyperparameters_to_search = rf_hyperparameters_to_search or default_rf_hyperparameters_to_search
    grid_search_settings = grid_search_settings or default_grid_search_settings

    xgb_model = XGBRegressor(enable_categorical=True, n_jobs=-1)
    rf_model = RandomForestRegressor(n_jobs=-1)

    models = {
        "MedianTarget": MedianTargetBaseline,
        "MeanTarget": MeanTargetBaseline,
        "LastRecord": lambda : LastRecordBaseline(temporal_order),
        "AdjustedLastRecord": lambda : AdjustedLastRecordBaseline(temporal_order),
        "RandomForest": grid_search_builder(rf_model, rf_hyperparameters_to_search, grid_search_settings),
        "XGBoost": grid_search_builder(xgb_model, xgb_hyperparameters_to_search, grid_search_settings),
    }

    training_dataset.save(os.path.join(savefolder, "training_dataset"))

    for name, model in models.items():
        tab_forecasting = TabularForecasting(training_dataset)
        tab_forecasting.apply_model(model, sleep=10 if name == "XGBoost" else 0)
        tab_forecasting.save(os.path.join(savefolder, f"forecasting_with_{name}"))


def two_stage_forecasting(h1_to_h2_folder: str, h1_h2_to_h3_folder: str, h1_to_h3_folder: str, savefolder: str):
    os.makedirs(savefolder, exist_ok=True)

    h1_to_h3_dataset = Dataset().load(os.path.join(h1_to_h3_folder, "training_dataset"))

    h1 = h1_to_h3_dataset.get_X()
    true_h3 = h1_to_h3_dataset.get_Y()

    true_h3.to_csv(os.path.join(savefolder, "true_Y.csv"))

    h1_h2_to_h3_dataset = Dataset().load(os.path.join(h1_h2_to_h3_folder, "training_dataset"))


    for model_folder in os.listdir(h1_to_h2_folder):
        if not model_folder.startswith("forecasting_with_"):
            continue
        model_name = model_folder.replace("forecasting_with_", "")
        if os.path.exists(os.path.join(h1_to_h2_folder, model_folder, "predicted_Y.csv")):
            pred_h2 = pd.read_csv(os.path.join(h1_to_h2_folder, model_folder, "predicted_Y.csv"), index_col=0)
        else:
            continue

        h1_pred_h2 = pd.concat([h1, pred_h2], axis=1)
        h1_pred_h2.to_csv(os.path.join(savefolder, f"H1_pred_H2_{model_name}.csv"))

        forecasting_h1_h2_to_h3 = TabularForecasting(h1_h2_to_h3_dataset)
        forecasting_h1_h2_to_h3.load(os.path.join(h1_h2_to_h3_folder, model_folder))

        pred_h3 = forecasting_h1_h2_to_h3.predict_kfolds(h1_pred_h2, verbose=True)
        pred_h3.to_csv(os.path.join(savefolder, f"predicted_Y_{model_name}.csv"))

        performance = pd.DataFrame()
        for parameter in pred_h3.columns:
            # true y is non-missing values
            true_y = true_h3[parameter].dropna()
            pred_y = pred_h3.loc[true_y.index, parameter]
            performance[parameter] = evaluate_prediction(true_y.values, pred_y.values, default_evaluation_metrics)

        time.sleep(5)
        performance.index.name = "Metric"
        performance.to_csv(os.path.join(savefolder, f"performance_{model_name}.csv"))


def two_stage_steen_forecasting(h1_to_h2_h3_folder: str, h1_h2_to_h3_folder: str, tab_h1_to_h2_folder: str, savefolder: str):
    os.makedirs(savefolder, exist_ok=True)

    h1_to_h2_h3_dataset = Dataset().load(os.path.join(h1_to_h2_h3_folder, "training_dataset"))

    h1 = h1_to_h2_h3_dataset.get_X()
    true_h3 = h1_to_h2_h3_dataset.get_Y()[["90_STEEN lost"]]

    true_h3.to_csv(os.path.join(savefolder, "true_Y.csv"))

    h1_h2_to_h3_dataset = Dataset().load(os.path.join(h1_h2_to_h3_folder, "training_dataset"))


    for model_folder in os.listdir(h1_to_h2_h3_folder):
        if not model_folder.startswith("forecasting_with_"):
            continue
        model_name = model_folder.replace("forecasting_with_", "")
        if os.path.exists(os.path.join(h1_to_h2_h3_folder, model_folder, "predicted_Y.csv")):
            pred_h2_steen = pd.read_csv(os.path.join(h1_to_h2_h3_folder, model_folder, "predicted_Y.csv"), index_col=0)[["80_STEEN lost"]]
        else:
            continue
        if os.path.exists(os.path.join(tab_h1_to_h2_folder, model_folder, "predicted_Y.csv")):
            pred_h2_tab = pd.read_csv(os.path.join(tab_h1_to_h2_folder, model_folder, "predicted_Y.csv"), index_col=0)
        else:
            continue

        h1_pred_h2 = pd.concat([h1, pred_h2_steen], axis=1).join(pred_h2_tab)
        h1_pred_h2.to_csv(os.path.join(savefolder, f"H1_pred_H2_{model_name}.csv"))

        forecasting_h1_h2_to_h3 = TabularForecasting(h1_h2_to_h3_dataset)
        forecasting_h1_h2_to_h3.load(os.path.join(h1_h2_to_h3_folder, model_folder))

        pred_h3 = forecasting_h1_h2_to_h3.predict_kfolds(h1_pred_h2, verbose=True)
        pred_h3.to_csv(os.path.join(savefolder, f"predicted_Y_{model_name}.csv"))

        performance = pd.DataFrame()
        for parameter in pred_h3.columns:
            # true y is non-missing values
            true_y = true_h3[parameter].dropna()
            pred_y = pred_h3.loc[true_y.index, parameter]
            performance[parameter] = evaluate_prediction(true_y.values, pred_y.values, default_evaluation_metrics)

        time.sleep(5)
        performance.index.name = "Metric"
        performance.to_csv(os.path.join(savefolder, f"performance_{model_name}.csv"))


def inference_pipeline(data: pd.DataFrame, forecasts_folder: str, savefolder: str, models: list[str] = None):
    dataset = Dataset().load(os.path.join(forecasts_folder, "training_dataset"))
    test_x = data[dataset.x_columns]
    test_y = data[dataset.y_columns]
    test_x = test_x.dropna()
    test_y = test_y.loc[test_x.index]

    os.makedirs(savefolder, exist_ok=True)
    test_x.to_csv(os.path.join(savefolder, "test_X.csv"))
    test_y.to_csv(os.path.join(savefolder, "true_Y.csv"))

    for model_folder in os.listdir(forecasts_folder):
        if not model_folder.startswith("forecasting_with_"):
            continue
        model_name = model_folder.replace("forecasting_with_", "")
        if models is not None and model_name not in models:
            continue
        forecasting = TabularForecasting(dataset)
        forecasting.load(os.path.join(forecasts_folder, model_folder))
        pred_y = forecasting.predict(test_x)
        pred_y.to_csv(os.path.join(savefolder, f"predicted_Y_{model_name}.csv"))

        performance = pd.DataFrame()
        for parameter in pred_y.columns:
            true_y = test_y[parameter].dropna()
            pred_y_ = pred_y.loc[true_y.index, parameter]
            performance[parameter] = evaluate_prediction(true_y.values, pred_y_.values, default_evaluation_metrics)
        performance.index.name = "Metric"
        performance.to_csv(os.path.join(savefolder, f"performance_{model_name}.csv"))


def two_stage_inference(h1_to_h2_inference_folder, h1_to_h3_inference_folder, h1_h2_to_h3_forecasts_folder: str, savefolder: str, models: list[str] = None):
    os.makedirs(savefolder, exist_ok=True)

    h1 = pd.read_csv(os.path.join(h1_to_h2_inference_folder, "test_X.csv"), index_col=0)
    true_h3 = pd.read_csv(os.path.join(h1_to_h3_inference_folder, "true_Y.csv"), index_col=0)
    true_h3.to_csv(os.path.join(savefolder, "true_Y.csv"))

    h1_h2_to_h3_dataset = Dataset()

    for model_folder in os.listdir(h1_h2_to_h3_forecasts_folder):
        if not model_folder.startswith("forecasting_with_"):
            continue
        model_name = model_folder.replace("forecasting_with_", "")
        if models is not None and model_name not in models:
            continue
        forecasting = TabularForecasting(h1_h2_to_h3_dataset)
        forecasting.load(os.path.join(h1_h2_to_h3_forecasts_folder, model_folder))
        if os.path.exists(os.path.join(h1_to_h2_inference_folder, f"predicted_Y_{model_name}.csv")):
            h2 = pd.read_csv(os.path.join(h1_to_h2_inference_folder, f"predicted_Y_{model_name}.csv"), index_col=0)
        else:
            continue
        h1_h2 = pd.concat([h1, h2], axis=1)
        h1_h2.to_csv(os.path.join(savefolder, f"H1_pred_H2_{model_name}.csv"))
        pred_h3 = forecasting.predict(h1_h2)
        pred_h3.to_csv(os.path.join(savefolder, f"predicted_Y_{model_name}.csv"))

        performance = pd.DataFrame()
        for parameter in pred_h3.columns:
            true_y = true_h3[parameter].dropna()
            pred_y = pred_h3.loc[true_y.index, parameter]
            performance[parameter] = evaluate_prediction(true_y.values, pred_y.values, default_evaluation_metrics)
        performance.index.name = "Metric"
        performance.to_csv(os.path.join(savefolder, f"performance_{model_name}.csv"))


def two_stage_steen_inference(h1_to_h2_h3_inference_folder, h1_h2_to_h3_forecasts_folder: str, tab_h1_to_h2_inference_folder: str, savefolder: str, models: list[str] = None):
    os.makedirs(savefolder, exist_ok=True)

    h1 = pd.read_csv(os.path.join(h1_to_h2_h3_inference_folder, "test_X.csv"), index_col=0)
    true_h3 = pd.read_csv(os.path.join(h1_to_h2_h3_inference_folder, "true_Y.csv"), index_col=0)[["90_STEEN lost"]]
    true_h3.to_csv(os.path.join(savefolder, "true_Y.csv"))

    h1_h2_to_h3_dataset = Dataset()

    for model_folder in os.listdir(h1_h2_to_h3_forecasts_folder):
        if not model_folder.startswith("forecasting_with_"):
            continue
        model_name = model_folder.replace("forecasting_with_", "")
        if models is not None and model_name not in models:
            continue
        forecasting = TabularForecasting(h1_h2_to_h3_dataset)
        forecasting.load(os.path.join(h1_h2_to_h3_forecasts_folder, model_folder))
        if os.path.exists(os.path.join(h1_to_h2_h3_inference_folder, f"predicted_Y_{model_name}.csv")):
            h2 = pd.read_csv(os.path.join(h1_to_h2_h3_inference_folder, f"predicted_Y_{model_name}.csv"), index_col=0)[["80_STEEN lost"]]
        else:
            continue
        if os.path.exists(os.path.join(tab_h1_to_h2_inference_folder, f"predicted_Y_{model_name}.csv")):
            h2_tab = pd.read_csv(os.path.join(tab_h1_to_h2_inference_folder, f"predicted_Y_{model_name}.csv"), index_col=0)
        else:
            continue
        h1_h2 = pd.concat([h1, h2], axis=1).join(h2_tab)
        h1_h2.to_csv(os.path.join(savefolder, f"H1_pred_H2_{model_name}.csv"))
        pred_h3 = forecasting.predict(h1_h2)
        pred_h3.to_csv(os.path.join(savefolder, f"predicted_Y_{model_name}.csv"))

        performance = pd.DataFrame()
        for parameter in pred_h3.columns:
            true_y = true_h3[parameter].dropna()
            pred_y = pred_h3.loc[true_y.index, parameter]
            performance[parameter] = evaluate_prediction(true_y.values, pred_y.values, default_evaluation_metrics)
        performance.index.name = "Metric"
        performance.to_csv(os.path.join(savefolder, f"performance_{model_name}.csv"))


def example_usage(savefolder: str = "Example Results"):
    features = ["A", "B", "C", "D"]
    prefixes = ["H1_", "H2_", "H3_"]
    column_names = [prefix + feature for prefix in prefixes for feature in features]
    n_samples = 200

    data = pd.DataFrame(index=range(1, n_samples+1), columns=column_names, dtype=float)
    for sample in data.index:
        for feature in features:
            a, b, c = np.random.normal(0, 2, 3)
            f = lambda x: a * x**2 + b * x + c
            for i, prefix in enumerate(prefixes):
                data.loc[sample, prefix + feature] = f(i+1) + np.random.normal(0, 0.1)
                if np.random.rand() < 0.02:
                    data.loc[sample, prefix + feature] = np.nan

    print(f"Data shape: {data.shape}")

    training_data = data.iloc[:int(n_samples * 0.8)]
    testing_data = data.iloc[int(n_samples * 0.8):]

    temp_order = temporal_order_from_patterns(data.columns, prefixes)
    print(f"Temporal order: {temp_order.orders}")

    dataset_h1_to_h2 = Dataset().create_from_df(
        data=training_data.drop(columns=[column for column in data.columns if 'H3_' in column]).copy(),
        target_columns=[column for column in data.columns if 'H2_' in column],
        drop_x_na=True,
        verbose=True
    )
    forecasting_pipeline(
        training_dataset=dataset_h1_to_h2,
        savefolder=os.path.join(savefolder, "H1_to_H2"),
        temporal_order=temp_order
    )

    dataset_h1_to_h3 = Dataset().create_from_df(
        data=training_data.drop(columns=[column for column in data.columns if 'H2_' in column]).copy(),
        target_columns=[column for column in data.columns if 'H3_' in column],
        drop_x_na=True,
        verbose=True
    )
    forecasting_pipeline(
        training_dataset=dataset_h1_to_h3,
        savefolder=os.path.join(savefolder, "H1_to_H3"),
        temporal_order=temp_order
    )

    dataset_h1_h2_to_h3 = Dataset().create_from_df(
        data=training_data.copy(),
        target_columns=[column for column in data.columns if 'H3_' in column],
        drop_x_na=True,
        verbose=True
    )
    forecasting_pipeline(
        training_dataset=dataset_h1_h2_to_h3,
        savefolder=os.path.join(savefolder, "H1_H2_to_H3"),
        temporal_order=temp_order
    )

    two_stage_forecasting(
        h1_to_h2_folder=os.path.join(savefolder, "H1_to_H2"),
        h1_h2_to_h3_folder=os.path.join(savefolder, "H1_H2_to_H3"),
        h1_to_h3_folder=os.path.join(savefolder, "H1_to_H3"),
        savefolder=os.path.join(savefolder, "H1_pred_H2_to_H3")
    )

    inference_pipeline(
        data=testing_data,
        forecasts_folder=os.path.join(savefolder, "H1_to_H2"),
        savefolder=os.path.join(savefolder, "H1_to_H2 inference")
    )

    inference_pipeline(
        data=testing_data,
        forecasts_folder=os.path.join(savefolder, "H1_to_H3"),
        savefolder=os.path.join(savefolder, "H1_to_H3 inference")
    )

    inference_pipeline(
        data=testing_data,
        forecasts_folder=os.path.join(savefolder, "H1_H2_to_H3"),
        savefolder=os.path.join(savefolder, "H1_H2_to_H3 inference")
    )

    two_stage_inference(
        h1_to_h2_inference_folder=os.path.join(savefolder, "H1_to_H2 inference"),
        h1_to_h3_inference_folder=os.path.join(savefolder, "H1_to_H3 inference"),
        h1_h2_to_h3_forecasts_folder=os.path.join(savefolder, "H1_H2_to_H3"),
        savefolder=os.path.join(savefolder, "H1_pred_H2_to_H3 inference")
    )


if __name__ == "__main__":
    example_usage()