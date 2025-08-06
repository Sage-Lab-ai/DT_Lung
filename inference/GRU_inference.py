import torch
import numpy as np
import pandas as pd
from GRU.util.baselines import average_of_tail
from GRU.forecasting_pipeline import param_idx
from GRU.GRU import GRU
import pathlib
from safetensors.torch import load_file

static_setups = ["A1F50_A2F50", "A1F50PA2F50_A3F50", "A1F50_A3F50"]
dynamic_setups = ["A1F50A2F50_A3F50"]
parameter_names = {
    "Dy_comp": "Dynamic Compliance (mL/cmH₂O)",
    "P_peak": "Peak Airway Pressure (cmH₂O)",
    "P_mean": "Mean Airway Pressure (cmH₂O)",
    "Ex_vol": "Expiratory Volume (mL)"
}   # Order must match the order of model input

device = torch.device("cpu")

def read_simulated_data(file_path):
    xl = pd.ExcelFile(file_path)

    simulated_data = {}
    sheet_map = {
        "A1F50": "1Hr Per-breath Data",
        "A2F50": "2Hr Per-breath Data",
        "A3F50": "3Hr Per-breath Data"
    }
    for key, sheet in sheet_map.items():
        if sheet in xl.sheet_names:
            simulated_data[key] = np.array(pd.read_excel(file_path, sheet_name=sheet)[list(parameter_names.values())]).reshape(1, 50, 4)
        else:
            simulated_data[key] = None
    return simulated_data

class TimeSeriesInference:

    def __init__(self, model_folder):
        self.model_folder = pathlib.Path(model_folder)

        self.a1 = None
        self.a2 = None
        self.a3 = None

        self.pred_a2 = None
        self.static_pred_a3 = None
        self.dynamic_pred_a3 = None

    def load_model(self, stage: str, param: str) -> torch.nn.Module:
        path = self.model_folder / "GRU" / stage / f"{param}.safetensors"
        state_dict = load_file(path)

        num_dynamic_feature = state_dict['gru.weight_ih_l0'].shape[1]
        gru_hidden_size = state_dict['gru.weight_hh_l0'].shape[1]
        output_sequence_length = state_dict['fcs.fc3.weight'].shape[0]

        model = GRU(
                    num_dynamic_feature=num_dynamic_feature, 
                    output_sequence_length=output_sequence_length, 
                    gru_hidden_size=gru_hidden_size, 
                    num_static_feature=0,
                    GRU_params={"num_layers": 1, "bidirectional": False}, 
                    mlp_hidden_size=[gru_hidden_size + 20, gru_hidden_size], 
                    mlp_activation="ReLU"
                    )
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model

    def load_input_data(self, dfs):
        if "1Hr Per-breath Data" in dfs:
            self.a1 = dfs["1Hr Per-breath Data"][list(parameter_names.values())]
        if "2Hr Per-breath Data" in dfs:
            self.a2 : pd.DataFrame= dfs["2Hr Per-breath Data"][list(parameter_names.values())]
            self.pred_a2 = self.a2.copy()
            self.pred_a2[:] = np.nan
        if "3Hr Per-breath Data" in dfs:
            self.a3 = dfs["3Hr Per-breath Data"][list(parameter_names.values())]
            self.static_pred_a3 = self.a3.copy()
            self.static_pred_a3[:] = np.nan
            self.dynamic_pred_a3 = self.a3.copy()
            self.dynamic_pred_a3[:] = np.nan

    def inference_all_parameters(self, input_x, model_setup):
        results = {}
        for param, param_name in parameter_names.items():
            model = self.load_model(model_setup, param)
            avg_of_x = average_of_tail(input_x)[:, :, param_idx[param]: param_idx[param] + 1]
            with torch.no_grad():
                x = torch.tensor(input_x, dtype=torch.float32).to(device, non_blocking=True)
                y_preds = model(x, None).detach().cpu().numpy()

            y_preds = y_preds + avg_of_x
            results[param_name] = y_preds.flatten()
        return results

    def static_inference(self):
        if self.a1 is None:
            raise ValueError("First hour time series data is missing, cannot perform static inference.")

        pred_a2_results = self.inference_all_parameters(self.a1.to_numpy().reshape(1, 50, 4), "A1F50_A2F50")
        for param, pred in pred_a2_results.items():
            self.pred_a2[param] = pred

        a1_pa2 = np.concatenate([self.a1.to_numpy().reshape(1, 50, 4), self.pred_a2.to_numpy().reshape(1, 50, 4)], axis=1)
        static_pred_a3_results = self.inference_all_parameters(a1_pa2, "A1F50A2F50_A3F50")

        for param, pred in static_pred_a3_results.items():
            self.static_pred_a3[param] = pred

    def dynamic_inference(self):
        if self.a1 is None or self.a2 is None:
            raise ValueError("First or second hour time series data is missing, cannot perform dynamic inference.")

        a1_a2 = np.concatenate([self.a1.to_numpy().reshape(1, 50, 4), self.a2.to_numpy().reshape(1, 50, 4)], axis=1)
        dynamic_pred_a3_results = self.inference_all_parameters(a1_a2, "A1F50A2F50_A3F50")
        for param, pred in dynamic_pred_a3_results.items():
            self.dynamic_pred_a3[param] = pred
