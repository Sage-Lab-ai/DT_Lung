import pandas as pd
import warnings
import pathlib
from huggingface_hub import snapshot_download

from inference.XGB_inference import XGBInference
from GRU.GRU import GRU
from inference.GRU_inference import TimeSeriesInference
from inference.reformat import OutputSheets


warnings.filterwarnings("ignore")

model_folder = pathlib.Path("Model")
model_folder.mkdir(exist_ok=True, parents=True)
data_folder = pathlib.Path("Data")
data_folder.mkdir(exist_ok=True, parents=True)
output_folder = pathlib.Path("Output")
output_folder.mkdir(exist_ok=True, parents=True)

snapshot_download("SageLabUHN/DT_Lung", local_dir=model_folder, local_dir_use_symlinks=False)
snapshot_download("SageLabUHN/DT_Lung_Demo_Data", repo_type="dataset", local_dir=data_folder, local_dir_use_symlinks=False)

inferences = {}
for demo_case in data_folder.glob("DT Lung Demo Case *.xlsx"):
    demo_case_name = demo_case.stem
    print(f"Processing {demo_case_name}...")

    print(f"\tLoading demo case data from '{demo_case}'...")
    demo_case_dfs = pd.read_excel(demo_case, sheet_name=None, index_col=0)

    xgb_inference = XGBInference(model_folder)
    xgb_inference.load_input_data(demo_case_dfs)

    print(f"\tRunning inference for {demo_case_name}...")
    xgb_inference.run()
    xgb_inference.get_pred_display()

    time_series_inference = TimeSeriesInference(model_folder)
    time_series_inference.load_input_data(demo_case_dfs)
    time_series_inference.static_inference()
    time_series_inference.dynamic_inference()

    save_path = output_folder / f"{demo_case_name} predictions.xlsx"
    print(f"\tSaving output to' {save_path}'...")
    with pd.ExcelWriter(save_path, mode='w') as writer:
        for sheet_name, df in xgb_inference.predictions_display.items():
            df.to_excel(writer, sheet_name=sheet_name)
        time_series_inference.pred_a2.to_excel(writer, sheet_name=OutputSheets.per_breath_h2)
        time_series_inference.static_pred_a3.to_excel(writer, sheet_name=OutputSheets.per_breath_h3_static)
        time_series_inference.dynamic_pred_a3.to_excel(writer, sheet_name=OutputSheets.per_breath_h3_dynamic)
    print()

    inferences[demo_case_name] = {
        "xgb": xgb_inference,
        "time_series": time_series_inference
    }

