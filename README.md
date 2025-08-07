
<p align="center">
  <img
    src="assets/icon.svg"
    alt="Respiratory Rate"
    width="72"
    height="72"
  />
</p>

<h1 align="center">Digital Twins of Ex Vivo Human Lungs</h1>

<p align="center">
  <a href="https://sage-lab-ai.github.io/DT_Lung/">
    <img alt="Project Page"
         src="https://img.shields.io/badge/Project-Page-blue" />
  </a>
  <a href="https://github.com/Sage-Lab-ai/DT_Lung">
    <img alt="GitHub Code"
         src="https://img.shields.io/badge/GitHub-Code-black?logo=github" />
  </a>
  <a href="https://colab.research.google.com/drive/1IrzccQ09mP5amxQTZy07n4LgLSC2r1uj?usp=sharing">
    <img alt="Colab: DT Lung Demo"
         src="https://img.shields.io/badge/Colab-DT%20Lung%20Demo-orange?logo=google-colab" />
  </a>
  <a href="https://dt-lung.streamlit.app/">
    <img alt="Streamlit Web App"
         src="https://img.shields.io/badge/Streamlit-Web%20App-red?logo=streamlit" />
  </a>
  <a href="https://huggingface.co/SageLabUHN/DT_Lung">
    <img alt="Hugging Face Model"
         src="https://img.shields.io/badge/HuggingFace-Model-yellow?logo=huggingface" />
  </a>
  <a href="https://huggingface.co/datasets/SageLabUHN/DT_Lung_Demo_Data">
    <img alt="Hugging Face Dataset"
         src="https://img.shields.io/badge/HuggingFace-Dataset-yellow?logo=huggingface" />
  </a>
</p>

This is the official repository for **Digital Twins of Ex Vivo Human Lungs**. <br />

---
## Table of Contents
- [Background and Motivation](#background-and-motivation)
- [Getting Started](#getting-started)
  - [🐍 Python users](#python-users)
  - [🐳 Running with Docker](#running-with-docker)
    - [Prerequisites](#prerequisites)
    - [For MacOS/Linux users](#for-macoslinux-users)
    - [For Windows users](#for-windows-users)
  - [📒 Google Colab Demo](#google-colab-demo)
  - [🌐 Web-based App (Code-free Deployment)](#web-based-app-for-non-technical-users)
- [DT Workflow](#dt-workflow)
  - [Code Structure](#code-structure)
  - [Naming convention](#naming-convention)
    - [GRU model training setups](#gru-model-training-setups)
    - [GRU model variables](#gru-model-variables)
    - [XGBoost model training setups](#xgboost-model-training-setups)
    - [XGBoost model variables](#xgboost-model-variables)
- [🤖 Inference](#inference)
- [🛠️ Troubleshooting Errors](#troubleshooting-errors)
  - [Web-app](#web-app)
  - [Docker](#docker)
  - [Colab](#colab)
- [📢 Stay up-to-date & 🐞 Report Issues](#stay-up-to-date-report-issues)


---
## Background and Motivation
Ex vivo lung perfusion (EVLP) is a cutting-edge platform that maintains isolated human lungs in a physiologically active state outside the body, enabling comprehensive functional assessment and targeted therapeutic interventions. The concept of a “digital twin” – a dynamic, high-fidelity computational replica of a physical system – is rapidly gaining traction in medicine for its ability to simulate complex biological processes in silico.

Leveraging these advances, we have curated the world’s largest annotated dataset of ex vivo human lungs. Using this rich dataset, our digital twin of ex vivo human lungs accurately forecasts over 75 key parameters of lung function (e.g., compliance, gas exchange, etc.) and has been validated as a robust digital control for preclinical therapeutics.

This repository contains the complete end-to-end pipeline, including model training scripts, inference modules, Docker containers, and an interactive web application, aiming to make the repository transparent, user-friendly, and conducive to further inspection, validation, and improvement by the broader research community.

**Ex vivo lung perfusion system**



https://github.com/user-attachments/assets/a3ea7f0d-7cdf-4c05-a6a0-160356bb3d40



---
## Getting Started

<a name="python-users"></a>
### 1. :snake: Python users
1. This DT model works with Python 3.12. Please make sure you have the correct version of Python installed before getting started.
2. Clone this repository:
   ````
   git clone https://github.com/Sage-Lab-ai/DT_Lung.git
   ````
4. All system requirements are listed in the requirements.txt file. To set up the environment, please run: <br />
   ````
   pip install -r requirements.txt
   ````
5. Run the `main.py` script <br />
   ````
   python main.py
   ````
   **Note**: Model weights and demo data are fetched automatically—no manual setup required.<br /> 

6. Once completed, your digital twins using demo data have been built! :white_check_mark:<br />
   DT results will be saved to `work_dir/DT_Lung/Output`. Please view results in the `Output` folder. 

<a name="running-with-docker"></a>
### 2. :whale: Running with Docker
#### Prerequisites:
[Docker](https://docs.docker.com/get-docker/) must be installed and running on your machine.<br />
All docker images are published on [Docker Hub](https://hub.docker.com/u/sagelabuhn)

This repository provides ready-to-use helper scripts to launch both the web application and the core main.py service in Docker containers.

  #### For MacOS/Linux users:
  
  Launch the web application:
   ````
   ./docker_run_app.sh
   ````

  Launch the main service:
   ````
   ./docker_run_main.sh
   ````
  To grant execute permission to Docker start scripts, please run:
   ````
   chmod +x docker_run_app.sh
   ````
  OR
  ````
   chmod +x docker_run_main.sh
   ````

  #### For Windows users
 
  Launch the web application: Double-click `docker_run_app.bat` in File Explorer.
  
  Run the main service: Double-click `docker_run_main.bat` in File Explorer.

<a name="google-colab-demo"></a>
### 3. :ledger: Google Colab Demo
[🚀 Launch in Colab](https://colab.research.google.com/drive/1IrzccQ09mP5amxQTZy07n4LgLSC2r1uj?usp=sharing) contains the demo with pre-written code cells on how to build a digital twin using our demo data: no code edit or local environment setup required. 

<a name="web-based-app-for-non-technical-users"></a>
### 4. :globe_with_meridians: Web-based App (Code-free Deployment)
The web-app offers an easy-to-follow user interface for seamless DT development that can be tailored to lung-specific conditions at the tip of your finger. <br />
[🚀 Launch DT Web-app](https://dt-lung.streamlit.app/)

#### App demo

<img width="1647" height="1138" alt="Screenshot 2025-08-06 at 3 42 28 PM" src="https://github.com/user-attachments/assets/3cc65041-85fe-4731-9931-53bb5d40da35" />

---
## DT Workflow
---

> 💡 **Did you know?** This model is trained on the largest dataset of its kind.

---

### Code Structure

The digital-twin pipeline in this repository is implemented using two core machine learning architectures: gated recurrent units (GRU) and XGBoost (XGB). The `GRU/` and `XGB/` directories each contain everything needed for model training, including data loading and preprocessing scripts, model architecture definitions and calibration, and utility functions that support the training pipeline. 

All inference scripts are located in the `inference/` folder, which provides code to load a trained model and generate predicted lung function parameters. For detailed instructions on running inference and building your own digital twin, please see the [Getting Started](#getting-started) section above and choose the inference method (command-line, Docker, Google Colab, or web-based app) that best suits you.  

```bash
project-root/
├── main.py                                  # Primary entry point for the DT_Lung pipeline
├── GRU                                      # Pipeline on GRU model training and calibration 
│   ├── __init__.py  
│   ├── scripts
│   ├── util
│   ├── EVLPMultivariateBreathDataset.py
│   ├── forecast_parameters.py
│   ├── forecast_parameters_w_pred.py
│   ├── forecasting_pipeline.py
│   ├── GRU.py
├── XGB                                      # Pipeline on XGBoost model training and calibration
│   ├── __init__.py
│   ├── BaselineModels.py
│   ├── Dataset.py
│   ├── TemporalOrder.py
│   ├── TabularForecasting.py
│   ├── pipelines.py
│   ├── inference.py
│   ├── forecasting_pipeline.py
│   ├── image_gridsearch_static.py   
│   ├── image_gridsearch_dynamic.py   
│   ├── image_train_static.py   
│   ├── image_train_dynamic.py   
│   ├── utils.py
├── inference
│   ├── __init__.py
│   ├── GRU_inference.py                     # Pipeline for all GRU-based model inference
│   ├── XGB_inference.py                     # Pipeline for all XGB-based model inference
│   ├── reformat.py                          # For enhancement on readability for general users
│   ├── visualization.py                     # Helper functions for DT visualization
├── Dockerfile                               # Defines the container image
├── docker_build.sh                          # Build Docker image
├── docker_run_app.sh                        # Running Docker container for web-app on MacOS/Linux
├── docker_run_app.bat                       # Running Docker container for web-app on Windows
├── docker_run_main.sh                       # Running Docker container for main service on MacOS/Linux
├── docker_run_main.bat                      # Running Docker container for main service on Windows
├── requirements.txt                         # System requirements
├── GLOSSARY.md                              # for units, data range references for users, acronyms, and domain terms
├── docs
├── assets
```

### Naming convention
This project follows a custom naming convention for model configurations and variable identifiers. Detailed explanations are provided below to enhance clarity and improve code readability.

#### GRU model training setups

A breath setup defines which breaths are included as input and output data to the model. They include:

`A1_A2` Setups for Static Digital Lung Forecasting:

- `A1F50_A2F50`
- `A1F50L50_A2F50`
- `N1L20A1F50L50_A2F50`

`A1PA2_A3` Setups for Static Digital Lung Forecasting:

- `A1F50PA2F50_A3F50`
- `A1F50L50PA2F50_A3F50`
- `N1L20A1F50L50PA2F50_A3F50`

`A1_A3` Setups for Static Digital Lung Forecasting:

- `A1F50_A3F50`
- `A1F50L50_A3F50`
- `N1L20A1F50L50_A3F50`

`A1A2_A3` Setups for Dynamic Digital Lung Forecasting: 

- `A1F50A2F50_A3F50`
- `A1F50L50A2F50_A3F50`
- `N1L20A1F50L50A2F50_A3F50`

Legend: `A = assessment period`, `N = normal breathing period`, `F = first breaths`, `L = last breaths` , `numbers = the number of breaths included`

*Note: everything before `_` is the input variables and everything after `_` is the target variable.*

#### GRU model variables

The variable defines which parameter will be forecasted. They include: 
- Dynamic Compliance (`Dy_Comp`)
- Peak Pressure (`P_peak`)
- Mean Pressure (`P_mean`)
- Expiratory Volume (`Ex_vol`)

#### XGBoost model training setups

`H1_to_H2`: Static digital lung forecasting (Forecast 2nd hour lung function using 1st hour baseline data)

`H1_to_H3`: Static digital lung forecasting (Forecast 3rd hour lung function using 1st hour baseline data)

`H1_predH2_to_H3`: Static digital lung forecasting (Forecast 3rd hour lung function using 1st hour baseline data and predicted 2nd hour data)

`H1_H2_to_H3`: Dynamic digital lung forecasting (Forecast 3rd hour lung function using 1st and 2nd hour obsereved data)

#### XGBoost model variables

Due to a large number of parameters, see [`GLOSSARY.md`](https://github.com/Sage-Lab-ai/DT_Lung/blob/main/GLOSSARY.md) for all abbreviation definitions.

---
<a name="inference"></a>
## 🤖 Inference (Create digital twins using your data📊🫁)

All trained models developed in this project can be found in our HuggingFace [Model Repository](https://huggingface.co/SageLabUHN/DT_Lung).
We also provide a [Demo Dataset](https://huggingface.co/datasets/SageLabUHN/DT_Lung_Demo_Data) on HuggingFace for users to try out our digital twin models.

We provide four distinct methods in the [Getting Started](#getting-started) section to run DT inference for creating digital twins of human lungs using either our demo data or your own data!

---
<a name="troubleshooting-errors"></a>
## 🛠️ Troubleshooting Errors

### Web-app

### Docker

### Colab 

---
<a name="stay-up-to-date-report-issues"></a>
## 📢 Stay up-to-date & 🐞 Report Issues

Interested in staying up-to-date with our work? Join our mailing list here: [Mailing List](https://forms.gle/saQ63UqN8REozCom8)

If you encounter any bugs or have ideas for improvement, please file an issue here: [Open a new issue](https://github.com/Sage-Lab-ai/DT_Lung/issues/new/choose)

## License
This dataset is released under the **Creative Commons Attribution‑NonCommercial‑ShareAlike 4.0 International (CC BY‑NC‑SA 4.0)** license. Commercial use is **prohibited**.













