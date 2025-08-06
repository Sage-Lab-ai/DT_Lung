import pandas as pd
import plotly.express as px
from .reformat import *


color_map = {
    "Observed": "grey",
    "Static Predicted": "#FD8008",
    "Dynamic Predicted": "#49548A"
}
dash_map = {
    "Observed": "solid",
    "Static Predicted": "dot",
    "Dynamic Predicted": "dash"
}


def hourly_all_features_line_plot(hourly_prediction: pd.DataFrame, wide: bool = True):
    hourly_prediction = hourly_prediction.drop(index=excluded_hourly_features_in_display, errors='ignore')
    n_features = len(hourly_prediction.index)
    observed = pd.DataFrame({
        "Hour": [1, 2, 3] * n_features,
        "Feature": hourly_prediction.index.repeat(3),
        "Value": hourly_prediction[[f"Observed {HourlyOrderMap.H1.label}", f"Observed {HourlyOrderMap.H2.label}", f"Observed {HourlyOrderMap.H3.label}"]].values.flatten(),
        "Type": "Observed"
    })
    static = pd.DataFrame({
        "Hour": [1, 2, 3] * n_features,
        "Feature": hourly_prediction.index.repeat(3),
        "Value": hourly_prediction[[f"Observed {HourlyOrderMap.H1.label}", f"Predicted {HourlyOrderMap.H2.label}", f"Static Predicted {HourlyOrderMap.H3.label}"]].values.flatten(),
        "Type": "Static Predicted"
    })
    dynamic = pd.DataFrame({
        "Hour": [1, 2, 3] * n_features,
        "Feature": hourly_prediction.index.repeat(3),
        "Value": hourly_prediction[[f"Observed {HourlyOrderMap.H1.label}", f"Observed {HourlyOrderMap.H2.label}", f"Dynamic Predicted {HourlyOrderMap.H3.label}"]].values.flatten(),
        "Type": "Dynamic Predicted"
    })
    df = pd.concat([observed, static, dynamic], ignore_index=True)
    df = df.dropna(subset=["Value"])

    y_padding = {
        # 'pMean': 2,
        'pPlat': 2,
        # 'pPeak': 2,
        'PAP': 2,
        'LAP': 2,
        'LA PO2': 20,
        'PA PO2': 10,
        'LA PCO2': 5,
        'PA PCO2': 5,
        'Cstat': 10,
        # 'Cdyn': 10,
        'LA Na+': 10,
        'LA Ca++': 0.5,
        'LA CL': 10,
        'LA K+': 1,
        'LA HCO3': 1,
        'LA BE': 5,
        'LA pH': 0.2,
        'LA Lact': 2,
        'LA Glu': 2,
        'STEEN lost': 50
    }
    y_boundary = {
        'LA BE': (None, -10),
        'STEEN lost': (0, None),
    }

    fig = px.line(
        df,
        x="Hour",
        y="Value",
        color="Type",
        color_discrete_map=color_map,
        line_dash="Type",
        line_dash_map=dash_map,
        facet_col="Feature",
        category_orders={"Feature": hourly_features_to_display},
        facet_col_wrap=6 if wide else 3,
        markers=True,
        title="Hourly Observations vs Predictions for Each Feature"
    )

    fig.for_each_yaxis(lambda y: y.update(matches=None, showticklabels=True))
    fig.for_each_annotation(lambda a: a.update(text=a.text.replace("Feature=", "")))
    fig.for_each_xaxis(lambda x: x.update(tickmode='array', tickvals=[1, 2, 3], showticklabels=True))

    fig.update_traces(opacity=0.7, line=dict(width=3), marker=dict(size=8))

    fig.update_layout(
        title_x=0.5,
        title_xanchor="center",
        title_y=0.98 if wide else 0.99,
        legend_title_text="",
        legend=dict(orientation="h", yanchor="bottom", y=1.05 if wide else 1.015, xanchor="center", x=0.5),
        margin=dict(t=100),
        height=800 if wide else 2000,
    )

    for i, annotation in enumerate(fig.layout.annotations):
        feature = annotation.text
        feature_data = df[df["Feature"] == feature]["Value"]
        padding = y_padding.get(HourlyMap.to_key(feature))
        if padding is None:
            continue
        lower, upper = y_boundary.get(HourlyMap.to_key(feature), (0, None))

        y_min = feature_data.min() - padding
        if lower is not None:
            y_min = max(y_min, lower)
        y_max = feature_data.max() + padding
        if upper is not None:
            y_max = min(y_max, upper)

        yaxis_name = f'yaxis{i+1}'
        if yaxis_name in fig.layout:
            fig.layout[yaxis_name].update(range=[y_min, y_max])

    return fig


def image_pc_line_plot(image_prediction: pd.DataFrame):
    n_features = len(image_prediction.index)
    observed = pd.DataFrame({
        "Hour": [1, 3] * n_features,
        "Feature": image_prediction.index.repeat(2),
        "Value": image_prediction[["Observed 1st Hour", "Observed 3rd Hour"]].values.flatten(),
        "Type": "Observed"
    })
    static = pd.DataFrame({
        "Hour": [1, 3] * n_features,
        "Feature": image_prediction.index.repeat(2),
        "Value": image_prediction[["Observed 1st Hour", "Static Predicted 3rd Hour"]].values.flatten(),
        "Type": "Static Predicted"
    })
    dynamic = pd.DataFrame({
        "Hour": [1, 3] * n_features,
        "Feature": image_prediction.index.repeat(2),
        "Value": image_prediction[["Observed 1st Hour", "Dynamic Predicted 3rd Hour"]].values.flatten(),
        "Type": "Dynamic Predicted"
    })
    df = pd.concat([observed, static, dynamic], ignore_index=True)
    df = df.dropna(subset=["Value"])

    fig = px.line(
        df,
        x="Hour",
        y="Value",
        color="Type",
        color_discrete_map=color_map,
        line_dash="Type",
        line_dash_map=dash_map,
        facet_col="Feature",
        category_orders={"Feature": list(ImagePCMap.all_labels())},
        facet_col_wrap=5,
        markers=True,
        title="Image PC Observations vs Predictions for Each Feature"
    )
    fig.for_each_yaxis(lambda y: y.update(matches=None, showticklabels=True))
    fig.for_each_annotation(lambda a: a.update(text=a.text.replace("Feature=", "")))
    fig.for_each_xaxis(lambda x: x.update(tickmode='array', tickvals=[1, 3], showticklabels=True))
    fig.update_traces(opacity=0.7, line=dict(width=3), marker=dict(size=8))
    fig.update_layout(
        title_x=0.5,
        title_xanchor="center",
        title_y=0.98,
        legend_title_text="",
        legend=dict(orientation="h", yanchor="bottom", y=1.12, xanchor="center", x=0.5),
        margin=dict(t=100),
        height=500,
    )
    return fig

def image_pc_scatter_plot(image_prediction: pd.DataFrame):
    static = pd.DataFrame({
        "Feature": image_prediction.index,
        "Observed 3rd Hour": image_prediction["Observed 3rd Hour"],
        "Predicted 3rd Hour": image_prediction["Static Predicted 3rd Hour"],
        "Type": "Static Predicted"
    })

    dynamic = pd.DataFrame({
        "Feature": image_prediction.index,
        "Observed 3rd Hour": image_prediction["Observed 3rd Hour"],
        "Predicted 3rd Hour": image_prediction["Dynamic Predicted 3rd Hour"],
        "Type": "Dynamic Predicted"
    })

    df = pd.concat([static, dynamic], ignore_index=True)
    df = df.dropna(subset=["Observed 3rd Hour", "Predicted 3rd Hour"])

    fig = px.scatter(
        df,
        x="Observed 3rd Hour",
        y="Predicted 3rd Hour",
        color="Type",
        color_discrete_map=color_map,
        title="Image PC Observations vs Predictions for 3rd Hour",
        labels={"Observed 3rd Hour": "Observed 3rd Hour", "Predicted 3rd Hour": "Predicted 3rd Hour"},
        # trendline="ols"
    )

    fig.update_traces(marker=dict(size=10, opacity=0.7), line=dict(width=2))
    fig.update_layout(
        title_x=0.5,
        title_xanchor="center",
        title_y=0.98,
        legend_title_text="",
        legend=dict(orientation="h", yanchor="bottom", y=1.12, xanchor="center", x=0.5),
        margin=dict(t=100),
        height=500,
    )

    return fig


def protein_line_plot(protein_prediction: pd.DataFrame, wide: bool = True):
    n_features = len(protein_prediction.index)
    observed = pd.DataFrame({
        "Minute": [60, 120, 180] * n_features,
        "Feature": protein_prediction.index.repeat(3),
        "Value": protein_prediction[[f"Observed {ProteinOrderMap.M60.label}", f"Observed {ProteinOrderMap.M120.label}", f"Observed {ProteinOrderMap.M180.label}"]].values.flatten(),
        "Type": "Observed"
    })
    static = pd.DataFrame({
        "Minute": [60, 120, 180] * n_features,
        "Feature": protein_prediction.index.repeat(3),
        "Value": protein_prediction[[f"Observed {ProteinOrderMap.M60.label}", f"Predicted {ProteinOrderMap.M120.label}", f"Static Predicted {ProteinOrderMap.M180.label}"]].values.flatten(),
        "Type": "Static Predicted"
    })
    dynamic = pd.DataFrame({
        "Minute": [60, 120, 180] * n_features,
        "Feature": protein_prediction.index.repeat(3),
        "Value": protein_prediction[[f"Observed {ProteinOrderMap.M60.label}", f"Observed {ProteinOrderMap.M120.label}", f"Dynamic Predicted {ProteinOrderMap.M180.label}"]].values.flatten(),
        "Type": "Dynamic Predicted"
    })
    df = pd.concat([observed, static, dynamic], ignore_index=True)
    df = df.dropna(subset=["Value"])

    fig = px.line(
        df,
        x="Minute",
        y="Value",
        color="Type",
        color_discrete_map=color_map,
        line_dash="Type",
        line_dash_map=dash_map,
        facet_col="Feature",
        category_orders={"Feature": list(ProteinMap.all_labels())},
        facet_col_wrap=4 if wide else 2,
        markers=True,
        title="Protein Observations vs Predictions for Each Feature"
    )
    fig.for_each_yaxis(lambda y: y.update(matches=None, showticklabels=True))
    fig.for_each_annotation(lambda a: a.update(text=a.text.replace("Feature=", "")))
    fig.for_each_xaxis(lambda x: x.update(tickmode='array', tickvals=[60, 120, 180], showticklabels=True))
    fig.update_traces(opacity=0.7, line=dict(width=3), marker=dict(size=8))
    fig.update_layout(
        title_x=0.5,
        title_xanchor="center",
        title_y=0.98 if wide else 0.99,
        legend_title_text="",
        legend=dict(orientation="h", yanchor="bottom", y=1.12 if wide else 1.05, xanchor="center", x=0.5),
        margin=dict(t=100),
        height=500 if wide else 1000,
    )
    return fig


def protein_line_plot_2(protein_prediction: pd.DataFrame):
    n_features = len(protein_prediction.index)
    observed = pd.DataFrame({
        "Minute": [60, 90, 110, 120, 130, 150, 180] * n_features,
        "Feature": protein_prediction.index.repeat(7),
        "Value": protein_prediction[["Observed 1st Hour", "Observed 90 Minutes", "Observed 110 Minutes", "Observed 2nd Hour", "Observed 130 Minutes", "Observed 150 Minutes", "Observed 3rd Hour"]].values.flatten(),
        "Type": "Observed"
    })
    static = pd.DataFrame({
        "Minute": [60, 120, 180] * n_features,
        "Feature": protein_prediction.index.repeat(3),
        "Value": protein_prediction[["Observed 1st Hour", "Predicted 2nd Hour", "Static Predicted 3rd Hour"]].values.flatten(),
        "Type": "Static Predicted"
    })
    dynamic = pd.DataFrame({
        "Minute": [60, 90, 110, 120, 180] * n_features,
        "Feature": protein_prediction.index.repeat(5),
        "Value": protein_prediction[["Observed 1st Hour", "Observed 90 Minutes", "Observed 110 Minutes", "Observed 2nd Hour", "Dynamic Predicted 3rd Hour"]].values.flatten(),
        "Type": "Dynamic Predicted"
    })
    df = pd.concat([observed, static, dynamic], ignore_index=True)
    df = df.dropna(subset=["Value"])

    fig = px.line(
        df,
        x="Minute",
        y="Value",
        color="Type",
        color_discrete_map=color_map,
        line_dash="Type",
        line_dash_map=dash_map,
        facet_col="Feature",
        category_orders={"Feature": list(ProteinMap.all_labels())},
        facet_col_wrap=4,
        markers=True,
        title="Protein Observations vs Predictions for Each Feature"
    )
    fig.for_each_yaxis(lambda y: y.update(matches=None, showticklabels=True))
    fig.for_each_annotation(lambda a: a.update(text=a.text.replace("Feature=", "")))
    fig.for_each_xaxis(lambda x: x.update(tickmode='array', tickvals=[60, 120, 180], showticklabels=True))
    fig.update_traces(opacity=0.7, line=dict(width=3), marker=dict(size=8))
    fig.update_layout(
        title_x=0.5,
        title_xanchor="center",
        title_y=0.98,
        legend_title_text="",
        legend=dict(orientation="h", yanchor="bottom", y=1.12, xanchor="center", x=0.5),
        margin=dict(t=100),
    )
    return fig


def transcriptomics_heatmap(transcriptomics_prediction: pd.DataFrame):
    rename_dict = {
        'HALLMARK_APOPTOSIS': 'Apoptosis',
        'HALLMARK_HYPOXIA': 'Hypoxia',
        'HALLMARK_IL2_STAT5_SIGNALING': 'IL2/STAT5 signaling',
        'HALLMARK_IL6_JAK_STAT3_SIGNALING': 'IL6/JAK/STAT3 signaling',
        'HALLMARK_INFLAMMATORY_RESPONSE': 'Inflammatory response',
        'HALLMARK_OXIDATIVE_PHOSPHORYLATION': 'Oxidative phosphorylation',
        'HALLMARK_P53_PATHWAY': 'Tumour protein p53 pathway',
        'HALLMARK_PI3K_AKT_MTOR_SIGNALING': 'PI3K/AKT/mTOR signaling',
        'HALLMARK_TGF_BETA_SIGNALING': 'TGF-β signaling',
        'HALLMARK_TNFA_SIGNALING_VIA_NFKB': 'TNF-α signaling pathway via NF-κB'
    }
    transcriptomics_prediction = transcriptomics_prediction.loc[list(rename_dict.keys()), [f"Observed {TranscriptomicsOrderMap.cit2.label}", f"Static Predicted {TranscriptomicsOrderMap.cit2.label}", f"Dynamic Predicted {TranscriptomicsOrderMap.cit2.label}"]]
    transcriptomics_prediction = transcriptomics_prediction.rename(index=rename_dict)
    transcriptomics_prediction = transcriptomics_prediction.sort_values(by=f"Observed {TranscriptomicsOrderMap.cit2.label}", ascending=False)
    fig = px.imshow(
        transcriptomics_prediction,
        color_continuous_scale='Viridis',
        aspect='auto',
        title="Transcriptomics Predictions Heatmap",
        labels={'y': 'Pathway', 'color': 'Enrichment Score'},
        text_auto="%d",
    )
    return fig


def transcriptomics_bar_plot(transcriptomics_prediction: pd.DataFrame):
    rename_dict = {
        'HALLMARK_APOPTOSIS': 'Apoptosis',
        'HALLMARK_HYPOXIA': 'Hypoxia',
        'HALLMARK_IL2_STAT5_SIGNALING': 'IL2/STAT5 signaling',
        'HALLMARK_IL6_JAK_STAT3_SIGNALING': 'IL6/JAK/STAT3 signaling',
        'HALLMARK_INFLAMMATORY_RESPONSE': 'Inflammatory response',
        'HALLMARK_OXIDATIVE_PHOSPHORYLATION': 'Oxidative phosphorylation',
        'HALLMARK_P53_PATHWAY': 'Tumour protein p53 pathway',
        'HALLMARK_PI3K_AKT_MTOR_SIGNALING': 'PI3K/AKT/mTOR signaling',
        'HALLMARK_TGF_BETA_SIGNALING': 'TGF-β signaling',
        'HALLMARK_TNFA_SIGNALING_VIA_NFKB': 'TNF-α signaling pathway via NF-κB'
    }
    transcriptomics_prediction = transcriptomics_prediction.loc[list(rename_dict.keys()), [f"Observed {TranscriptomicsOrderMap.cit2.label}", f"Static Predicted {TranscriptomicsOrderMap.cit2.label}", f"Dynamic Predicted {TranscriptomicsOrderMap.cit2.label}"]]
    transcriptomics_prediction = transcriptomics_prediction.rename(index=rename_dict)
    transcriptomics_prediction = transcriptomics_prediction.sort_values(by=f"Observed {TranscriptomicsOrderMap.cit2.label}", ascending=False)
    fig = px.bar(
        transcriptomics_prediction,
        barmode='group',
        title="Transcriptomics Predictions Bar Plot",
        labels={'variable': 'Prediction Type', 'value': 'Enrichment Score', 'index': 'Pathway'},
    )
    return fig


def timeseries_plot(a1, true_a2, true_a3, pred_a2, static_pred_a3, dynamic_pred_a3):
    figs = []
    paddings = {
        PerBreathParameterMap.Dy_comp.label: 60,
        PerBreathParameterMap.P_peak.label: 13,
        PerBreathParameterMap.P_mean.label: 8,
        PerBreathParameterMap.Ex_vol.label: 80
    }
    for parameter in PerBreathParameterMap.all_labels():
        parameter_no_unit = parameter.split(" (")[0]
        param_a1 = pd.DataFrame({
            "Breath": np.arange(50),
            "Value": a1[parameter].to_numpy().flatten(),
            "Type": "Observed",
            "Hour": "1Hr"
        })

        param_a2 = pd.DataFrame({
            "Breath": np.arange(50),
            "Value": true_a2[parameter].to_numpy().flatten(),
            "Type": "Observed",
            "Hour": "2Hr"
        })

        param_a3 = pd.DataFrame({
            "Breath": np.arange(50),
            "Value": true_a3[parameter].to_numpy().flatten(),
            "Type": "Observed",
            "Hour": "3Hr"
        })

        param_pred_a2 = pd.DataFrame({
            "Breath": np.arange(50),
            "Value": pred_a2[parameter].to_numpy().flatten(),
            "Type": "Static Predicted",
            "Hour": "2Hr"
        })

        param_static_a3 = pd.DataFrame({
            "Breath": np.arange(50),
            "Value": static_pred_a3[parameter].to_numpy().flatten(),
            "Type": "Static Predicted",
            "Hour": "3Hr"
        })

        param_dynamic_a3 = pd.DataFrame({
            "Breath": np.arange(50),
            "Value": dynamic_pred_a3[parameter].to_numpy().flatten(),
            "Type": "Dynamic Predicted",
            "Hour": "3Hr"
        })

        # Combine all
        param_df = pd.concat([param_a1, param_a2, param_a3, param_pred_a2, param_static_a3, param_dynamic_a3], ignore_index=True)
        param_df = param_df.dropna(subset=["Value"])

        # Create line plot
        fig = px.line(
            param_df,
            x="Breath",
            y="Value",
            color="Type",
            color_discrete_map=color_map,
            line_dash="Type",
            line_dash_map=dash_map,
            facet_col="Hour",
            category_orders={"Hour": ["1Hr", "2Hr", "3Hr"]},
            labels={"Value": parameter, "Breath": "Breath"},
            title=f"{parameter_no_unit} Per Breath Over Hours"
        )

        # Final touches
        fig.update_layout(
            title_x=0.5,
            title_xanchor="center",
            title_y=0.98,
            legend_title_text="",
            legend=dict(orientation="h", yanchor="bottom", y=1.1, xanchor="center", x=0.5),
            margin=dict(t=100),
            height=500,
        )
        fig.for_each_annotation(lambda a: a.update(text=a.text.replace("Hour=", "") + f" {parameter_no_unit}"))
        y_lim = max(param_df["Value"].min() - paddings[parameter], 0), param_df["Value"].max() + paddings[parameter]
        fig.update_yaxes(range=y_lim)
        # set x-lim to -10 to 60
        fig.update_xaxes(range=[-5, 55], tickmode='array', tickvals=[0, 10, 20, 30, 40, 50], showticklabels=True)
        figs.append(fig)

    return figs