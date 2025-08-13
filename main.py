import pandas as pd
import warnings
import pathlib
from huggingface_hub import snapshot_download
import time
from inference.XGB_inference import XGBInference
from GRU.GRU import GRU
from inference.GRU_inference import TimeSeriesInference
from inference.reformat import OutputSheets
import tarfile

warnings.filterwarnings("ignore")

model_folder = pathlib.Path("Model")
model_folder.mkdir(exist_ok=True, parents=True)
data_folder = pathlib.Path("Data")
data_folder.mkdir(exist_ok=True, parents=True)
output_folder = pathlib.Path("Output")
output_folder.mkdir(exist_ok=True, parents=True)

print("Downloading models and demo data from huggingface...")
try:
    snapshot_download("SageLabUHN/DT_Lung_Demo_Data", repo_type="dataset", local_dir=data_folder, max_workers=1, allow_patterns=["*.xlsx", "*.md"])
    time.sleep(5)
    snapshot_download("SageLabUHN/DT_Lung", local_dir=model_folder, max_workers=4, allow_patterns=["*.tar.gz", "*.md"])
except Exception as e:
    print(f"Failed to download models or demo data due to Internet issue. Please try again in a few seconds.")
    exit(1)

tar_files = [file for file in model_folder.glob("*.tar.gz")]
for tar_file in tar_files:
    with tarfile.open(tar_file, "r:gz") as tar:
        tar.extractall(path=model_folder)

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
    time_series_inference.get_pred_display()

    save_path = output_folder / f"{demo_case_name} predictions.xlsx"
    print(f"\tSaving output to' {save_path}'...")
    with pd.ExcelWriter(save_path, mode='w') as writer:
        for sheet_name, df in xgb_inference.predictions_display.items():
            df.to_excel(writer, sheet_name=sheet_name)
        for sheet_name, df in time_series_inference.pred_display.items():
            df.to_excel(writer, sheet_name=sheet_name)
    print()

    inferences[demo_case_name] = {
        "xgb": xgb_inference,
        "time_series": time_series_inference
    }

print("\n** All results saved to the DT_Lung/Output folder.")
