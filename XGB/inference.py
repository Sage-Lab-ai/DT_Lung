from .pipelines import *


def hourly_inference_new_cases(new_data, model_folder, save_folder, suffix=" inference", models=None):
    inference_pipeline(
        data=new_data,
        forecasts_folder=os.path.join(model_folder, "H1_to_H2"),
        savefolder=os.path.join(save_folder, "H1_to_H2" + suffix),
        models=models
    )

    inference_pipeline(
        data=new_data,
        forecasts_folder=os.path.join(model_folder, "H1_to_H3"),
        savefolder=os.path.join(save_folder, "H1_to_H3" + suffix),
        models=models
    )

    inference_pipeline(
        data=new_data,
        forecasts_folder=os.path.join(model_folder, "H1_H2_to_H3"),
        savefolder=os.path.join(save_folder, "H1_H2_to_H3" + suffix),
        models=models
    )

    two_stage_inference(
        h1_to_h2_inference_folder=os.path.join(save_folder, "H1_to_H2" + suffix),
        h1_to_h3_inference_folder=os.path.join(save_folder, "H1_to_H3" + suffix),
        h1_h2_to_h3_forecasts_folder=os.path.join(model_folder, "H1_H2_to_H3"),
        savefolder=os.path.join(save_folder, "H1_pred_H2_to_H3" + suffix),
        models=models
    )

    inference_pipeline(
        data=new_data,
        forecasts_folder=os.path.join(model_folder, 'STEEN_H1_to_H2_H3'),
        savefolder=os.path.join(save_folder, 'STEEN_H1_to_H2_H3' + suffix),
        models=models
    )

    inference_pipeline(
        data=new_data,
        forecasts_folder=os.path.join(model_folder, 'STEEN_H1_H2_to_H3'),
        savefolder=os.path.join(save_folder, 'STEEN_H1_H2_to_H3' + suffix),
        models=models
    )

    two_stage_steen_inference(
        h1_to_h2_h3_inference_folder=os.path.join(save_folder, "STEEN_H1_to_H2_H3" + suffix),
        h1_h2_to_h3_forecasts_folder=os.path.join(model_folder, "STEEN_H1_H2_to_H3"),
        tab_h1_to_h2_inference_folder=os.path.join(save_folder, "H1_to_H2" + suffix),
        savefolder=os.path.join(save_folder, "STEEN_H1_pred_H2_to_H3" + suffix),
        models=models
    )


def protein_inference_new_cases_dynamic(new_data, model_folder, save_folder, suffix=" inference", models=None):
    inference_pipeline(
        data=new_data,
        forecasts_folder=os.path.join(model_folder, 'H1_to_H2'),
        savefolder=os.path.join(save_folder, 'H1_to_H2' + suffix),
        models=models
    )

    inference_pipeline(
        data=new_data,
        forecasts_folder=os.path.join(model_folder, 'H1_to_H3'),
        savefolder=os.path.join(save_folder, 'H1_to_H3' + suffix),
        models=models
    )

    inference_pipeline(
        data=new_data,
        forecasts_folder=os.path.join(model_folder, 'H1_H2_to_H3'),
        savefolder=os.path.join(save_folder, 'H1_H2_to_H3' + suffix),
        models=models
    )


def protein_inference_new_cases_static(new_data, model_folder, save_folder, suffix=" inference", models=None):
    inference_pipeline(
        data=new_data,
        forecasts_folder=os.path.join(model_folder, 'H1_pred_H2_to_H3'),
        savefolder=os.path.join(save_folder, 'H1_pred_H2_to_H3' + suffix),
        models=models
    )


def transcriptomics_inference_new_cases_dynamic(new_data, model_folder, save_folder, suffix=" inference", models=None):
    inference_pipeline(
        data=new_data,
        forecasts_folder=os.path.join(model_folder, 'dynamic_forecasting'),
        savefolder=os.path.join(save_folder, 'dynamic_forecasting' + suffix),
        models=models
    )

def transcriptomics_inference_new_cases_static(new_data, model_folder, save_folder, suffix=" inference", models=None):
    inference_pipeline(
        data=new_data,
        forecasts_folder=os.path.join(model_folder, 'static_forecasting'),
        savefolder=os.path.join(save_folder, 'static_forecasting' + suffix),
        models=models
    )
