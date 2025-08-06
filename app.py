from inference.reformat import *
from inference.XGB_inference import XGBInference
from GRU.GRU import GRU
from inference.GRU_inference import TimeSeriesInference
from inference.visualization import *
import warnings
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from pathlib import Path
from huggingface_hub import snapshot_download
import time


warnings.filterwarnings("ignore")


def load_excel(file_path: Path):
    return load_excel_cache(file_path, file_path.stat().st_mtime)

@st.cache_data
def load_excel_cache(file_path, stat):
    return pd.read_excel(file_path, sheet_name=None, index_col=0)


def load_excel_binary(file_path):
    return load_excel_binary_cache(file_path, file_path.stat().st_mtime)

@st.cache_data
def load_excel_binary_cache(file_path, stat):
    with open(file_path, "rb") as f:
        return f.read()

def retry_snapshot_download(max_attempts=5, delay=3, *args, **kwargs):
    for attempt in range(max_attempts):
        try:
            return snapshot_download(*args, **kwargs)
        except Exception as e:
            if attempt < max_attempts - 1:
                time.sleep(delay)
            else:
                raise e
    return None

def download_huggingface(data_folder, model_folder):
    model_repo_id = "SageLabUHN/DT_Lung"
    data_repo_id = "SageLabUHN/DT_Lung_Demo_Data"
    with st.spinner(f"Downloading dataset from huggingface.co/datasets/{data_repo_id}"):
        retry_snapshot_download(repo_id=data_repo_id, repo_type="dataset", local_dir=data_folder, local_dir_use_symlinks=False)
    st.success(f"Successfully downloaded dataset from huggingface.co/datasets/{data_repo_id}")
    with st.spinner(f"Downloading model from huggingface.co/{model_repo_id}"):
        retry_snapshot_download(repo_id=model_repo_id, local_dir=model_folder, local_dir_use_symlinks=False)
    st.success(f"Successfully downloaded model from huggingface.co/{model_repo_id}")

def check_missing(data, tolerance=1.0):
    good = "✅ All Good"
    missing = "❌ {0:.1%} Missing"
    some_missing = "⚠️ {0:.1%} Missing"
    null_rate = data.isnull().to_numpy().flatten().mean()
    return good if null_rate == 0 else some_missing.format(null_rate) if null_rate < tolerance else missing.format(null_rate)

def check_modality_missing(dfs):
    hourly_missing = pd.Series(name=InputSheets.hourly_lung_function, index=list(HourlyOrderMap.all_labels()), dtype=str)
    for label in HourlyOrderMap.all_labels():
        hourly_missing[label] = check_missing(dfs[InputSheets.hourly_lung_function].loc[hourly_features_to_display, label])
    hourly_1_status = hourly_missing[HourlyOrderMap.H1.label]
    hourly_2_status = hourly_missing[HourlyOrderMap.H2.label]

    lung_image_missing = pd.Series(name=InputSheets.lung_image, index=ImagePCOrderMap.all_labels(), dtype=str)
    for label in ImagePCOrderMap.all_labels():
        lung_image_missing[label] = check_missing(dfs[InputSheets.lung_image][label])
    pc_1_status = lung_image_missing[ImagePCOrderMap.H1.label]
    pc_3_status = lung_image_missing[ImagePCOrderMap.H3.label]

    protein_missing = pd.Series(name=InputSheets.protein,
                                index=[f"{ProteinOrderMap.M60.label} to {ProteinOrderMap.M110.label}",
                                       f"{ProteinOrderMap.M120.label} to {ProteinOrderMap.M150.label}",
                                       ProteinOrderMap.M180.label],
                                dtype=str)
    protein_1_status = check_missing(dfs[InputSheets.protein][[ProteinOrderMap.M60.label, ProteinOrderMap.M90.label, ProteinOrderMap.M110.label]])
    protein_2_status = check_missing(dfs[InputSheets.protein][[ProteinOrderMap.M120.label, ProteinOrderMap.M130.label, ProteinOrderMap.M150.label]])
    protein_missing[f"{ProteinOrderMap.M60.label} to {ProteinOrderMap.M110.label}"] = protein_1_status
    protein_missing[f"{ProteinOrderMap.M120.label} to {ProteinOrderMap.M150.label}"] = protein_2_status
    protein_missing[ProteinOrderMap.M180.label] = check_missing(dfs[InputSheets.protein][ProteinOrderMap.M180.label])

    cit_missing = pd.Series(name=InputSheets.transcriptomics, index=TranscriptomicsOrderMap.all_labels(), dtype=str)
    for label in TranscriptomicsOrderMap.all_labels():
        cit_missing[label] = check_missing(dfs[InputSheets.transcriptomics][label])
    cit1_status = cit_missing[TranscriptomicsOrderMap.cit1.label]

    ts_missing = pd.Series(name="Per-breath Data", index=["1Hr", "2Hr", "3Hr"], dtype=str)
    ts_missing["1Hr"] = check_missing(dfs[InputSheets.per_breath_h1], tolerance=0)
    ts_missing["2Hr"] = check_missing(dfs[InputSheets.per_breath_h2], tolerance=0)
    ts_missing["3Hr"] = check_missing(dfs[InputSheets.per_breath_h3], tolerance=0)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.dataframe(hourly_missing)
    col2.dataframe(lung_image_missing)
    col3.dataframe(protein_missing)
    col4.dataframe(cit_missing)
    col5.dataframe(ts_missing)

    xgb_missing = any(status[0] != "✅" for status in (hourly_1_status, hourly_2_status, pc_1_status, pc_3_status, protein_1_status, protein_2_status, cit1_status))
    static_gru = ts_missing["1Hr"][0] == "✅"
    dynamic_gru = static_gru and ts_missing["2Hr"][0] == "✅"

    if xgb_missing:
        st.warning("DT result will be affected due to missing values in input.")
    if not static_gru:
        st.warning("Static GRU inference will NOT be performed due to missing 1Hr per-breath data.")
    if not dynamic_gru:
        st.warning("Dynamic GRU inference will NOT be performed due to missing 2Hr per-breath data.")

    return static_gru, dynamic_gru

def run_xgb_inference(model_folder, demo_dfs):
    xgb_inference = XGBInference(model_folder)
    xgb_inference.load_input_data(demo_dfs)
    start = time.time()
    with st.spinner("Running Hourly Lung Function Inference..."):
        xgb_inference.hourly_dynamic_inference()
        xgb_inference.hourly_static_inference()
    end = time.time()
    st.success(f"Hourly Lung Function Inference completed in {end - start:.2f} seconds.")
    start = time.time()
    with st.spinner("Running Lung Image Inference..."):
        xgb_inference.image_pc_dynamic_inference()
        xgb_inference.image_pc_static_inference()
    end = time.time()
    st.success(f"Lung Image Inference completed in {end - start:.2f} seconds.")
    start = time.time()
    with st.spinner("Running Protein Inference..."):
        xgb_inference.protein_dynamic_inference()
        xgb_inference.protein_static_inference()
    end = time.time()
    st.success(f"Protein Inference completed in {end - start:.2f} seconds.")
    start = time.time()
    with st.spinner("Running Transcriptomics Inference..."):
        xgb_inference.transcriptomics_dynamic_inference()
        xgb_inference.transcriptomics_static_inference()
        xgb_inference.get_pred_display()
    end = time.time()
    st.success(f"Transcriptomics Inference completed in {end - start:.2f} seconds.")
    return xgb_inference

def run_gru_inference(model_folder, demo_dfs, static_gru=True, dynamic_gru=True):
    time_series_inference = TimeSeriesInference(model_folder)
    time_series_inference.load_input_data(demo_dfs)
    if static_gru:
        start = time.time()
        with st.spinner("Running Static Time Series Inference"):
            time_series_inference.static_inference()
        end = time.time()
        st.success(f"Static Time Series Inference completed in {end - start:.2f} seconds.")
    if dynamic_gru:
        start = time.time()
        with st.spinner("Running Dynamic Time Series Inference"):
            time_series_inference.dynamic_inference()
        end = time.time()
        st.success(f"Dynamic Time Series Inference completed in {end - start:.2f} seconds.")
    return time_series_inference

def main():

    st.set_page_config(
        page_title="EVLP DT",
        page_icon=":material/respiratory_rate:",
        initial_sidebar_state="expanded",
        layout="wide",
    )

    if "huggingface_downloaded" not in st.session_state:
        st.session_state["huggingface_downloaded"] = False
    if "data_mode" not in st.session_state:
        st.session_state["data_mode"] = "demo"

    data_folder = Path("Data")
    data_folder.mkdir(parents=True, exist_ok=True)
    model_folder = Path("Model")
    model_folder.mkdir(parents=True, exist_ok=True)
    output_folder = Path("Output")
    output_folder.mkdir(parents=True, exist_ok=True)

    demo_case_prefix = "DT Lung Demo Case "

    st.title(":material/respiratory_rate: Digital Twin of Ex-Vivo Human Lungs ")

    st.info("❤️ Welcome to the Digital Twin of Ex-Vivo Human Lungs App! 👋 This app is best viewed in a desktop browser in light mode. 🖥️ To change the theme, navigate to the top right corner of the page and select '⋮' > 'Settings'.")

    st.subheader("Step 0: Download Models and Data")

    try:
        if not st.session_state["huggingface_downloaded"]:
            download_huggingface(data_folder, model_folder)
            st.session_state["huggingface_downloaded"] = True
        else:
            st.success("Models and data already downloaded. You can redownload them if needed.")
    except Exception as e:
        st.error(f"Failed to download from huggingface due to internet issue. Please try again in a few seconds.")
        st.session_state["huggingface_downloaded"] = False
    finally:
        redownload_huggingface = st.button(
            label="Redownload Models and Data",
            icon=":material/refresh:",
            use_container_width=True,
        )
        if redownload_huggingface:
            download_huggingface(data_folder, model_folder)

    st.subheader("Step 1: Prepare Data")

    st.info("Two options available: Please select if you want to **Use Demo Data** :bar_chart: OR **Use Your Own Data** :file_folder: to get started!")

    col1, col2 = st.columns(2, border=True)

    with col1:
        st.button(
            "**✔ Use Demo Data**" if st.session_state["data_mode"] == "demo" else "Use Demo Data",
            type="primary" if st.session_state["data_mode"] == "demo" else "secondary",
            use_container_width=True,
            on_click=lambda: st.session_state.update(data_mode="demo"),
        )
        demo_files = data_folder.glob(demo_case_prefix + "*")
        demo_names = [file.stem for file in demo_files if file.is_file()]
        demo_names.sort()
        selected_demo_case = st.selectbox(
            label="Select a Demo Case",
            options=demo_names,
            index=0,
            disabled=st.session_state["data_mode"] != "demo",
        )

    with col2:
        st.button(
            "✔ Use Your Own Data" if st.session_state["data_mode"] == "custom" else "Use Your Own Data",
            type="primary" if st.session_state["data_mode"] == "custom" else "secondary",
            use_container_width=True,
            on_click=lambda: st.session_state.update(data_mode="custom"),
        )

        st.download_button(
            label="Click Here to Download the Template (Excel File)",
            data=load_excel_binary(data_folder / "DT Lung Demo Template.xlsx"),
            file_name="DT Lung Demo Template.xlsx",
            mime="application/vnd.ms-excel",
            type='tertiary',
            disabled=st.session_state["data_mode"] != "custom",
        )
        uploaded_case = st.file_uploader(
            label="Upload Your Own Data (must be an Excel file with the same format as the template)",
            type=["xlsx"],
            accept_multiple_files=False,
            disabled=st.session_state["data_mode"] != "custom",
        )

    if st.session_state["data_mode"] == "demo":
        case_dfs = load_excel(data_folder / f"{selected_demo_case}.xlsx")
        selected_case = selected_demo_case
    else:
        case_dfs = load_excel_cache(uploaded_case, None) if uploaded_case else None
        selected_case = uploaded_case.name.replace(".xlsx", "") if uploaded_case else None

    if case_dfs is None:
        st.info("Upload an Excel to see the data preview and run inference.")
        return

    case_dfs[InputSheets.hourly_lung_function] = case_dfs[InputSheets.hourly_lung_function].reindex(list(HourlyMap.all_labels()))
    hourly_display_df = case_dfs[InputSheets.hourly_lung_function]
    image_pc_display_df = case_dfs[InputSheets.lung_image]
    protein_display_df = case_dfs[InputSheets.protein]
    transcriptomics_display_df = case_dfs[InputSheets.transcriptomics]
    time_series_display_dfs = [
        case_dfs[InputSheets.per_breath_h1],
        case_dfs[InputSheets.per_breath_h2],
        case_dfs[InputSheets.per_breath_h3]
    ]

    with (st.expander(f"Data Preview for {selected_case}", expanded=True)):
        (
            hourly_display_tab,
            image_pc_display_tab,
            protein_display_tab,
            transcriptomics_display_tab,
            time_series_a1_display_tab,
            time_series_a2_display_tab,
            time_series_a3_display_tab,
        ) = st.tabs([
            "Hourly Lung Function Data",
            "Lung Image Data",
            "Protein Data",
            "Transcriptomics Data",
            "1st Hour per-breath Data",
            "2nd Hour per-breath Data",
            "3rd Hour per-breath Data"
        ])
        hourly_display_tab.dataframe(hourly_display_df.loc[hourly_features_to_display])
        image_pc_display_tab.dataframe(image_pc_display_df)
        protein_display_tab.dataframe(protein_display_df)
        transcriptomics_display_tab.dataframe(transcriptomics_display_df)
        time_series_a1_display_tab.dataframe(time_series_display_dfs[0])
        time_series_a2_display_tab.dataframe(time_series_display_dfs[1])
        time_series_a3_display_tab.dataframe(time_series_display_dfs[2])

    st.subheader("Step 2: Run Inference")

    static_gru, dynamic_gru = check_modality_missing(case_dfs)

    prediction_save_path = output_folder / f"{selected_case} predictions.xlsx"

    predictions_display = None
    if prediction_save_path.exists():
        st.info(f"Predictions for {selected_case} already exist. You can view the results below or re-run the inference.")
        predictions_display = load_excel(prediction_save_path)

    run_inference = st.button(
        label="Run Inference",
        icon=":material/play_arrow:",
        use_container_width=True
    )

    if run_inference:
        xgb_inference = run_xgb_inference(model_folder, case_dfs)
        gru_inference = run_gru_inference(model_folder, case_dfs, static_gru, dynamic_gru)
        st.success("✅ All Inference completed successfully!")
        with pd.ExcelWriter(prediction_save_path, mode='w') as writer:
            for sheet_name, df in xgb_inference.predictions_display.items():
                df.to_excel(writer, sheet_name=sheet_name)
            gru_inference.pred_a2.to_excel(writer, sheet_name=OutputSheets.per_breath_h2)
            gru_inference.static_pred_a3.to_excel(writer, sheet_name=OutputSheets.per_breath_h3_static)
            gru_inference.dynamic_pred_a3.to_excel(writer, sheet_name=OutputSheets.per_breath_h3_dynamic)
        predictions_display = load_excel(prediction_save_path)

    st.subheader("Step 3: View Results")
    if predictions_display is not None:
        with st.container(border=True):
            (
                hourly_pred_tab,
                image_pc_pred_tab,
                protein_pred_tab,
                transcriptomics_pred_tab,
                time_series_pred_tab,
            ) = st.tabs([
                "Hourly Lung Function Prediction",
                "Lung X-ray Image Prediction",
                "Protein Prediction",
                "Transcriptomics Prediction",
                "Per-breath Predictions",
            ])

        hourly_pred_tab.dataframe(predictions_display["Hourly Lung Function Prediction"].loc[hourly_features_to_display])
        hourly_pred_tab.plotly_chart(
            hourly_all_features_line_plot(predictions_display["Hourly Lung Function Prediction"]),
            use_container_width=True,
        )

        image_pc_pred_tab.dataframe(predictions_display["Lung X-ray Image Prediction"])
        # image_pc_pred_tab.plotly_chart(
        #     image_pc_scatter_plot(predictions_display["Lung X-ray Image Prediction"]),
        #     use_container_width=True,
        # )

        st.write("**Note:** For a detailed description of the methodology for deriving image-based features, please refer to our previous publication. [:material/article: **Link to Paper**](https://doi.org/10.1038/s41746-024-01260-z)")

        protein_pred_tab.dataframe(predictions_display["Protein Prediction"])
        protein_pred_tab.plotly_chart(
            protein_line_plot(predictions_display["Protein Prediction"]),
            use_container_width=True,
            key="protein_line_plot_1"
        )
        # protein_pred_tab.plotly_chart(
        #     protein_line_plot_2(predictions_display["Protein Prediction"]),
        #     use_container_width=True,
        #     key="protein_line_plot_2"
        # )

        transcriptomics_pred_tab.dataframe(predictions_display["Transcriptomics Prediction"])
        transcriptomics_pred_tab.plotly_chart(
            transcriptomics_heatmap(predictions_display["Transcriptomics Prediction"]),
            use_container_width=True,
        )
        transcriptomics_pred_tab.plotly_chart(
            transcriptomics_bar_plot(predictions_display["Transcriptomics Prediction"]),
            use_container_width=True,
        )

        time_series_pred_tab.markdown("**2Hr Per-breath Prediction**")
        time_series_pred_tab.dataframe(predictions_display["2Hr Per-breath Prediction"])
        col1, col2 = time_series_pred_tab.columns(2)
        col1.markdown("**3Hr Per-breath Static Prediction**")
        col1.dataframe(predictions_display["3Hr Per-breath Static"])
        col2.markdown("**3Hr Per-breath Dynamic Prediction**")
        col2.dataframe(predictions_display["3Hr Per-breath Dynamic"])

        figs = timeseries_plot(
            case_dfs[InputSheets.per_breath_h1],
            case_dfs[InputSheets.per_breath_h2],
            case_dfs[InputSheets.per_breath_h3],
            predictions_display[OutputSheets.per_breath_h2],
            predictions_display[OutputSheets.per_breath_h3_static],
            predictions_display[OutputSheets.per_breath_h3_dynamic],
        )
        col1, col2 = time_series_pred_tab.columns(2)
        col1.plotly_chart(figs[0], use_container_width=True)
        col1.plotly_chart(figs[1], use_container_width=True)
        col2.plotly_chart(figs[2], use_container_width=True)
        col2.plotly_chart(figs[3], use_container_width=True)

    else:
        st.warning("No predictions available to view. Please run the inference in **Step 2** first.")

    st.subheader("Step 4: Download Predictions")
    if prediction_save_path.exists():
        saved_predictions = load_excel_binary(prediction_save_path)

        st.download_button(
            label="Download Predictions",
            data=saved_predictions,
            file_name=f"{selected_case} predictions.xlsx",
            mime="application/vnd.ms-excel",
            disabled=saved_predictions is None,
            use_container_width=True
        )
    else:
        st.warning("No predictions available to download. Please run the inference in **Step 2** first.")


if __name__ == "__main__":
    main()
