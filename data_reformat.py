import streamlit as st
import numpy as np
import pandas as pd
import pathlib
# import openpyxl

from inference.reformat import *


@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

def choose_id_column(data, key):
    id_col = st.selectbox(
        label=f"Choose Case ID column",
        options=list(data.columns),
        index=0,
        key=f"id_col_{key}"
    )
    data.set_index(id_col, inplace=True, drop=True)

def choose_id_column_timeseries(data, key):
    id_col = st.selectbox(
        label=f"Choose Case ID column",
        options=list(data.columns),
        index=0,
        key=f"id_col_{key}"
    )
    param_col = st.selectbox(
        label=f"Choose Parameter column",
        options=list(data.columns),
        index=1,
        key=f"param_col_{key}"
    )
    data.set_index([id_col, param_col], inplace=True, drop=True)

def preview_data(data):
    col1, col2, col3 = st.columns(3)
    col1.metric(label="Shape", value=str(data.shape), border=True)
    col2.metric(label="Missing", value=data.isnull().sum().sum(), border=True)
    col3.metric(label="Memory Usage", value=f"{data.memory_usage(deep=True).sum() / 1024:.2f} KB", border=True)
    st.dataframe(data, use_container_width=True)

def get_csv_input(uploader_label, key, ts=False):
    with st.container(border=True):
        uploaded_file = st.file_uploader(label=uploader_label, type=["csv"], key=f"uploader {key}")
        if uploaded_file is not None:
            data = load_data(uploaded_file)
            columns = list(data.columns)
            with st.expander("Columns to use", expanded=False):
                cols_to_use = st.pills(
                    label="Select columns to use",
                    label_visibility="collapsed",
                    selection_mode="multi",
                    options=columns,
                    default=columns,
                    key=f"cols_to_use {key}"
                )
                data = data[cols_to_use]
            if ts:
                choose_id_column_timeseries(data, key)
            else:
                choose_id_column(data, key)
            preview_data(data)
            return data
        else:
            return None


def compare_series_mismatches(
        s1: pd.Series,
        s2: pd.Series,
        ignore_type: bool = True,
        s1_name: str = "s1",
        s2_name: str = "s2"
) -> pd.DataFrame | None:
    s1_aligned, s2_aligned = s1.align(s2)
    if ignore_type:
        s1_comp = s1_aligned.astype(float)
        s2_comp = s2_aligned.astype(float)
    else:
        s1_comp = s1_aligned
        s2_comp = s2_aligned
    mismatch_mask = ~(s1_comp.eq(s2_comp) | (s1_comp.isna() & s2_comp.isna()))
    if mismatch_mask.any():
        return pd.DataFrame({
            s1_name: s1_aligned[mismatch_mask],
            s2_name: s2_aligned[mismatch_mask]
        })
    else:
        return None


def main():

    st.set_page_config(
        page_title="EVLP DT Data Reformatting Tool",
        page_icon="🧰",
        layout="wide"
    )

    st.title("🧰EVLP DT Data Reformatting Tool")

    st.subheader("Step 1: Upload Input Data")

    (
        hourly_input_tab,
        edema_input_tab,
        pc_1hr_input_tab,
        pc_3hr_input_tab,
        protein_input_tab,
        cit1_input_tab,
        cit2_input_tab2,
        ts_input_tab,
    ) = st.tabs([
        "Lung Function Hourly Data",
        "Lung Edema Data",
        "Lung X-ray 1Hr Data",
        "Lung X-ray 3Hr Data",
        "Protein Data",
        "Transcriptomics Baseline Data",
        "Transcriptomics Target Data",
        "Per-breath Time-series Data"
    ])

    with hourly_input_tab:
        hourly_input_df = get_csv_input(
            uploader_label=f"Upload Lung Function Hourly CSV File",
            key="hourly_input",
        )
    with edema_input_tab:
        edema_input_df = get_csv_input(
            uploader_label=f"Upload Lung Edema Data CSV File",
            key="edema_input",
        )
    with pc_1hr_input_tab:
        pc_1hr_input_df = get_csv_input(
            uploader_label=f"Upload Lung X-ray 1Hr Data CSV File",
            key="pc_1hr_input",
        )
    with pc_3hr_input_tab:
        pc_3hr_input_df = get_csv_input(
            uploader_label=f"Upload Lung X-ray 3hr Data CSV File",
            key="pc_3hr_input",
        )
    with protein_input_tab:
        protein_input_df = get_csv_input(
            uploader_label=f"Upload Protein Data CSV File",
            key="protein_input",
        )
    with cit1_input_tab:
        cit1_input_df = get_csv_input(
            uploader_label=f"Upload Transcriptomics Baseline Data CSV File",
            key="cit1_input",
        )
    with cit2_input_tab2:
        cit2_input_df = get_csv_input(
            uploader_label=f"Upload Transcriptomics Target Data CSV File",
            key="cit2_input",
        )
    with ts_input_tab:
        timeseries_input_df = get_csv_input(
            uploader_label=f"Upload Per-breath Time-series Data CSV File",
            key="ts_input",
            ts=True
        )

    cases = []
    if hourly_input_df is not None:
        cases = hourly_input_df.index
    elif edema_input_df is not None:
        cases = edema_input_df.index
    elif pc_1hr_input_df is not None:
        cases = pc_1hr_input_df.index
    elif pc_3hr_input_df is not None:
        cases = pc_3hr_input_df.index
    elif protein_input_df is not None:
        cases = protein_input_df.index
    elif cit1_input_df is not None:
        cases = cit1_input_df.index
    elif cit2_input_df is not None:
        cases = cit2_input_df.index
    elif timeseries_input_df is not None:
        cases = timeseries_input_df.index.get_level_values(0).unique()

    st.subheader("Step 2: Review Display Data")
    case_name_prefix = "Demo Case "
    selected_case_name = st.selectbox(
        "Select a case for display",
        options=(case_name_prefix + str(case) for case in cases),
        index=0,
    )
    if selected_case_name is not None:
        selected_case_id = int(selected_case_name.replace(case_name_prefix, ""))
    (
        hourly_display_tab,
        pc_display_tab,
        protein_display_tab,
        transcriptomics_display_tab,
        ts_h1_display_tab,
        ts_h2_display_tab,
        ts_h3_display_tab,
    ) = st.tabs([
        InputSheets.hourly_lung_function,
        InputSheets.lung_image,
        InputSheets.protein,
        InputSheets.transcriptomics,
        InputSheets.per_breath_h1,
        InputSheets.per_breath_h2,
        InputSheets.per_breath_h3,
    ])

    hourly_display_df = None
    pc_display_df = None
    protein_display_df = None
    transcriptomics_display_df = None

    if hourly_input_df is None:
        hourly_display_tab.warning("No Lung Function Hourly Data Uploaded.")
    if edema_input_df is None:
        hourly_display_tab.warning("No Lung Edema Data Uploaded.")
    if hourly_input_df is not None and edema_input_df is not None:
        hourly_display_df = HourlyTranslator.to_display_table(hourly_input_df.loc[selected_case_id], edema_input_df.loc[selected_case_id])
        hourly_display_tab.dataframe(hourly_display_df, use_container_width=True)
        hourly_calculated_delta = HourlyTranslator.compute_derived_features(hourly_display_df)
        hourly_display_tab.dataframe(hourly_calculated_delta, use_container_width=True)

    if pc_1hr_input_df is None:
        pc_display_tab.warning("No Lung X-ray 1Hr Data Uploaded.")
    if pc_3hr_input_df is None:
        pc_display_tab.warning("No Lung X-ray 3Hr Data Uploaded.")
    if pc_1hr_input_df is not None and pc_3hr_input_df is not None:
        pc_display_df = ImagePCTranslator.to_display_table(pc_1hr_input_df.loc[selected_case_id], pc_3hr_input_df.loc[selected_case_id])
        pc_display_tab.dataframe(pc_display_df, use_container_width=True)

    if protein_input_df is None:
        protein_display_tab.warning("No Protein Data Uploaded.")
    if protein_input_df is not None:
        protein_display_df = ProteinTranslator.to_display_table(protein_input_df.loc[selected_case_id])
        protein_display_tab.dataframe(protein_display_df, use_container_width=True)
        protein_slope_df = ProteinTranslator.compute_slopes(protein_display_df)
        protein_display_tab.dataframe(protein_slope_df, use_container_width=True)

    if cit1_input_df is None:
        transcriptomics_display_tab.warning("No Transcriptomics Baseline Data Uploaded.")
    if cit2_input_df is None:
        transcriptomics_display_tab.warning("No Transcriptomics Target Data Uploaded.")
    if cit1_input_df is not None and cit2_input_df is not None:
        transcriptomics_display_df = TranscriptomicsTranslator.to_display_table(
            cit1_input_df.loc[selected_case_id],
            cit2_input_df.loc[selected_case_id]
        )
        transcriptomics_display_tab.dataframe(transcriptomics_display_df, use_container_width=True)

    if timeseries_input_df is None:
        ts_h1_display_tab.warning("No Per-breath Time-series Data Uploaded.")
        ts_h2_display_tab.warning("No Per-breath Time-series Data Uploaded.")
        ts_h3_display_tab.warning("No Per-breath Time-series Data Uploaded.")
    else:
        timeseries_display_dfs = PerBreathTranslator.to_display_table(timeseries_input_df.loc[selected_case_id])
        ts_h1_display_tab.dataframe(timeseries_display_dfs["A1"], use_container_width=True)
        ts_h2_display_tab.dataframe(timeseries_display_dfs["A2"], use_container_width=True)
        ts_h3_display_tab.dataframe(timeseries_display_dfs["A3"], use_container_width=True)

    st.subheader("Step 3: Convert to Model Input")
    (
        hourly_model_input_tab,
        pc_model_input_tab,
        protein_model_input_tab,
        cit_model_input_tab,
    ) = st.tabs([
        "Lung Function Hourly Data",
        "Lung X-ray Data",
        "Protein Data",
        "Transcriptomics Data",
    ])
    if hourly_display_df is not None:
        hourly_with_calculated_display_df = pd.concat([hourly_display_df, hourly_calculated_delta], axis=0)
        hourly_model_input_df = pd.DataFrame([HourlyTranslator.to_input_table(hourly_with_calculated_display_df)], index=[selected_case_id])
        hourly_model_input_tab.dataframe(hourly_model_input_df)
        hourly_mismatches = compare_series_mismatches(
            hourly_model_input_df.loc[selected_case_id, hourly_input_df.columns],
            hourly_input_df.loc[selected_case_id],
            s1_name="Converted Data", s2_name="Original Data",
        )
        if hourly_mismatches is None:
            hourly_model_input_tab.success("Reverted Lung Function Hourly Data matches the original input data.")
        else:
            hourly_model_input_tab.error("Reverted Lung Function Hourly Data does not match the original input data.")
            hourly_model_input_tab.dataframe(hourly_mismatches, use_container_width=True)
        edema_mismatches = compare_series_mismatches(
            hourly_model_input_df.loc[selected_case_id, edema_input_df.columns],
            edema_input_df.loc[selected_case_id],
            s1_name="Converted Data", s2_name="Original Data",
        )
        if edema_mismatches is None:
            hourly_model_input_tab.success("Reverted Lung Edema Data matches the original input data.")
        else:
            hourly_model_input_tab.error("Reverted Lung Edema Data does not match the original input data.")
            hourly_model_input_tab.dataframe(edema_mismatches, use_container_width=True)
    if pc_display_df is not None:
        pc_model_input_h1_df, pc_model_input_h3_df = ImagePCTranslator.to_input_table(pc_display_df)
        pc_model_input_h1_df = pd.DataFrame([pc_model_input_h1_df], index=[selected_case_id])
        pc_model_input_h3_df = pd.DataFrame([pc_model_input_h3_df], index=[selected_case_id])
        pc_model_input_tab.dataframe(pc_model_input_h1_df, use_container_width=True)
        pc_model_input_tab.dataframe(pc_model_input_h3_df, use_container_width=True)
        pc_h1_mismatches = compare_series_mismatches(
            pc_model_input_h1_df.loc[selected_case_id, pc_1hr_input_df.columns],
            pc_1hr_input_df.loc[selected_case_id],
            s1_name="Converted Data", s2_name="Original Data",
        )
        if pc_h1_mismatches is None:
            pc_model_input_tab.success("Reverted Lung X-ray 1Hr Data matches the original input data.")
        else:
            pc_model_input_tab.error("Reverted Lung X-ray 1Hr Data does not match the original input data.")
            pc_model_input_tab.dataframe(pc_h1_mismatches, use_container_width=True)
        pc_h3_mismatches = compare_series_mismatches(
            pc_model_input_h3_df.loc[selected_case_id, pc_3hr_input_df.columns],
            pc_3hr_input_df.loc[selected_case_id],
            s1_name="Converted Data", s2_name="Original Data",
        )
        if pc_h3_mismatches is None:
            pc_model_input_tab.success("Reverted Lung X-ray 3Hr Data matches the original input data.")
        else:
            pc_model_input_tab.error("Reverted Lung X-ray 3Hr Data does not match the original input data.")
            pc_model_input_tab.dataframe(pc_h3_mismatches, use_container_width=True)

    if protein_display_df is not None:
        protein_model_input_df = pd.DataFrame([ProteinTranslator.to_input_table(protein_display_df)], index=[selected_case_id])
        protein_slope_input_df = pd.DataFrame([ProteinTranslator.slopes_to_input_table(protein_slope_df)], index=[selected_case_id])
        protein_with_slope_model_input_df = pd.concat([protein_model_input_df, protein_slope_input_df], axis=1)
        protein_model_input_tab.dataframe(protein_with_slope_model_input_df, use_container_width=True)
        protein_mismatches = compare_series_mismatches(
            protein_with_slope_model_input_df.loc[selected_case_id, protein_input_df.columns],
            protein_input_df.loc[selected_case_id],
            s1_name="Converted Data", s2_name="Original Data",
        )
        if protein_mismatches is None:
            protein_model_input_tab.success("Reverted Protein Data matches the original input data.")
        else:
            protein_model_input_tab.error("Reverted Protein Data does not match the original input data.")
            protein_model_input_tab.dataframe(protein_mismatches, use_container_width=True)

    if transcriptomics_display_df is not None:
        transcriptomics_model_input_df = pd.DataFrame([TranscriptomicsTranslator.to_input_table(transcriptomics_display_df)], index=[selected_case_id])
        cit_model_input_tab.dataframe(transcriptomics_model_input_df, use_container_width=True)
        cit1_mismatches = compare_series_mismatches(
            transcriptomics_model_input_df.loc[selected_case_id, cit1_input_df.columns],
            cit1_input_df.loc[selected_case_id],
            s1_name="Converted Data", s2_name="Original Data",
        )
        if cit1_mismatches is None:
            cit_model_input_tab.success("Reverted Transcriptomics Baseline Data matches the original input data.")
        else:
            cit_model_input_tab.error("Reverted Transcriptomics Baseline Data does not match the original input data.")
            cit_model_input_tab.dataframe(cit1_mismatches, use_container_width=True)
        cit2_mismatches = compare_series_mismatches(
            transcriptomics_model_input_df.loc[selected_case_id, cit2_input_df.columns],
            cit2_input_df.loc[selected_case_id],
            s1_name="Converted Data", s2_name="Original Data",
        )
        if cit2_mismatches is None:
            cit_model_input_tab.success("Reverted Transcriptomics Target Data matches the original input data.")
        else:
            cit_model_input_tab.error("Reverted Transcriptomics Target Data does not match the original input data.")
            cit_model_input_tab.dataframe(cit2_mismatches, use_container_width=True)

    st.subheader("Step 4: Download Display Data")
    all_exist = True
    if hourly_input_df is None:
        st.warning("No Lung Function Hourly Data Uploaded.")
        all_exist = False
    if edema_input_df is None:
        st.warning("No Lung Edema Data Uploaded.")
        all_exist = False
    if pc_1hr_input_df is None:
        st.warning("No Lung X-ray 1Hr Data Uploaded.")
        all_exist = False
    if pc_3hr_input_df is None:
        st.warning("No Lung X-ray 3Hr Data Uploaded.")
        all_exist = False
    if protein_input_df is None:
        st.warning("No Protein Data Uploaded.")
        all_exist = False
    if cit1_input_df is None:
        st.warning("No Transcriptomics Baseline Data Uploaded.")
        all_exist = False
    if cit2_input_df is None:
        st.warning("No Transcriptomics Target Data Uploaded.")
        all_exist = False

    col1, col2 = st.columns(2, vertical_alignment='bottom')

    save_folder_name = col1.text_input(
        label="Folder name to save Excel files",
        value="Display Data",
        placeholder="Enter folder name"
    )

    download_all_display_button = col2.button(
        label="Download per-case Excel for all cases",
        disabled=not all_exist or save_folder_name is None,
        use_container_width=True
    )

    if download_all_display_button:
        save_folder = pathlib.Path(save_folder_name)
        save_folder.mkdir(parents=True, exist_ok=True)
        for case in cases:
            case_name = f"DT Lung Demo Case {case}"
            with pd.ExcelWriter(save_folder / f"{case_name}.xlsx", mode='w') as writer:
                hourly_display_df = HourlyTranslator.to_display_table(
                    hourly_input_df.loc[case],
                    edema_input_df.loc[case]
                )
                pc_display_df = ImagePCTranslator.to_display_table(
                    pc_1hr_input_df.loc[case],
                    pc_3hr_input_df.loc[case]
                )
                protein_display_df = ProteinTranslator.to_display_table(protein_input_df.loc[case])

                transcriptomics_display_df = TranscriptomicsTranslator.to_display_table(
                    cit1_input_df.loc[case],
                    cit2_input_df.loc[case]
                )
                timeseries_display_dfs = PerBreathTranslator.to_display_table(timeseries_input_df.loc[case])
                hourly_display_df.to_excel(writer, sheet_name=InputSheets.hourly_lung_function)
                pc_display_df.to_excel(writer, sheet_name=InputSheets.lung_image)
                protein_display_df.to_excel(writer, sheet_name=InputSheets.protein)
                transcriptomics_display_df.to_excel(writer, sheet_name=InputSheets.transcriptomics)
                timeseries_display_dfs["A1"].to_excel(writer, sheet_name=InputSheets.per_breath_h1)
                timeseries_display_dfs["A2"].to_excel(writer, sheet_name=InputSheets.per_breath_h2)
                timeseries_display_dfs["A3"].to_excel(writer, sheet_name=InputSheets.per_breath_h3)
        template_hourly_display_df = pd.DataFrame(
            index=hourly_features_to_display,
            columns=list(HourlyOrderMap.all_labels())
        )
        template_pc_display_df = pd.DataFrame(
            index=list(ImagePCMap.all_labels()),
            columns=list(ImagePCOrderMap.all_labels())
        )
        template_protein_display_df = pd.DataFrame(
            index=list(ProteinMap.all_labels()),
            columns=list(ProteinOrderMap.all_labels())
        )
        template_transcriptomics_display_df = pd.DataFrame(
            index=transcriptomics_display_df.index,
            columns=list(TranscriptomicsOrderMap.all_labels())
        )
        template_timeseries_display_df = pd.DataFrame(
            index=timeseries_display_dfs["A1"].index,
            columns= list(PerBreathParameterMap.all_labels())
        )
        with pd.ExcelWriter(save_folder / "Display Data Template.xlsx", mode='w') as writer:
            template_hourly_display_df.to_excel(writer, sheet_name=InputSheets.hourly_lung_function)
            template_pc_display_df.to_excel(writer, sheet_name=InputSheets.lung_image)
            template_protein_display_df.to_excel(writer, sheet_name=InputSheets.protein)
            template_transcriptomics_display_df.to_excel(writer, sheet_name=InputSheets.transcriptomics)
            template_timeseries_display_df.to_excel(writer, sheet_name=InputSheets.per_breath_h1)
            template_timeseries_display_df.to_excel(writer, sheet_name=InputSheets.per_breath_h2)
            template_timeseries_display_df.to_excel(writer, sheet_name=InputSheets.per_breath_h3)
        st.success(f"All display data saved in {save_folder_name} folder.")


if __name__ == '__main__':
    main()


