<h1 align="center">Digital Twins of Ex Vivo Human Lungs</h1>

<p align="center">
  <a href="https://your-project-page-url">
    <img alt="Project Page"
         src="https://img.shields.io/badge/Project-Page-blue" />
  </a>
  <a href="[https://github.com/your-repo](https://github.com/Sage-Lab-ai/DT_Lung)">
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
  <a href="https://huggingface.co/SageLabUHN/DT_Lung/tree/main">
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
## Updates

Interested in staying up-to-date with our work? Join our mailing list here: [Mailing List](https://forms.gle/saQ63UqN8REozCom8)

---
## Table of Contents
- [Overview and Motivation](#overview-and-motivation)
- [Getting Started](#getting-started)
  - [🐍 Python users](#python-users)
  - [🐳 Running with Docker](#running-with-docker)
    - [For MacOS/Linux users](#for-macoslinux-users)
    - [For Windows users](#for-windows-users)
  - [📒 Google Colab Demo](#google-colab-demo)
  - [🌐 Web-based app (for non-technical users)](#web-based-app-for-non-technical-users)
- [DT Workflow](#dt-workflow)
  - [Code structure](#code-structure)
  - [Naming convention](#naming-convention)
    - [GRU model training setups](#gru-model-training-setups)
    - [GRU model variables](#gru-model-variables)
    - [XGBoost model training setups](#xgboost-model-training-setups)
    - [XGBoost model variables](#xgboost-model-variables)

---
## Background and Motivation
Here we provide the code files used to develop the digital twin model of human ex vivo lungs using the world's largest dataset on human ex vivo lung function. The digital lung model can accurately forecast ex vivo human lung function.

Ex vivo lung perfusion system

https://github.com/user-attachments/assets/44dfc3fb-8aa8-4313-8ab7-e8e937f0e057

---
## Getting Started

<a name="python-users"></a>
### 1. :snake: Python users
1. This DT model works with Python 3.12. Please make sure you have the correct version of Python installed before getting started.
2. Clone this repository:
   ````
   git clone https://github.com/Sage-Lab-ai/DT_Lung.git
   ````
4. Download all trained models from huggingface:
   ````
   from huggingface_hub import snapshot_download
   snapshot_download(repo_id= 'sagelab/DT_Lung', local_dir='./DT_Lung/Model', local_dir_use_symlinks=False)
   ````
   **Make sure the `Model` folder is in `work_dir/DT_Lung/` for proper execution of DT inference.**
6. Download demo data from huggingface:
   ````
   from huggingface_hub import snapshot_download
   snapshot_download(repo_id="SageLabUHN/DT_Lung_Demo_Data",
                     repo_type="dataset",
                     local_dir="work_dir/DT_Lung/Data/",
                      local_dir_use_symlinks=False)
   ````
   **Make sure the `Data` folder is in `work_dir/DT_Lung/` for proper execution of DT inference using demo data.**
8. All system requirements are listed in the requirements.txt file. To set up the environment, please run: <br />
   ````
   pip install -r requirements.txt
   ````
10. Run the `main.py` file
11. Review DT results in the `work_dir/DT_Lung/Output` (Your digital twins have been built! :white_check_mark:)

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

  #### For Windows users
 
  Launch the web application: Double-click `docker_run_app.bat` in File Explorer.
  
  Run the main service: Double-click `docker_run_main.bat` in File Explorer.

<a name="google-colab-demo"></a>
### 3. :ledger: Google Colab Demo
[🚀 Launch in Colab](https://colab.research.google.com/drive/1IrzccQ09mP5amxQTZy07n4LgLSC2r1uj?usp=sharing) contains the demo with pre-written code on how to build a digital twin using our demo data: no code edit or local environment setup required. 

<a name="web-based-app-for-non-technical-users"></a>
### 4. :globe_with_meridians: Web-based App (Code-free deployment)
The web-app offers an easy-to-follow user interface for seamless DT development that can be tailored to lung-specific conditions at the tip of your finger. <br />
[🚀 Launch DT Web-app](link) - Link to be updated

#### App demo
***Add a gif image of the app interface here***
Add recommended ranges of parameters, show this on the app.

---
## DT Workflow
---

> 💡 **Did you know?** This model is trained on the largest dataset of its kind.

---

### Code structure 

The DT model is built using two machine learning model architectures: gated recurrent unit (GRU) and XGBoost (XGB). In this repository, code files are organized and presented as shown below: <br />
The code files for model training are organized under the corresponding folders `GRU` and `XGB`. Under each folder, there are..... <br />
The inference files are included under the inference folder. If you wish to run inference to build your own DT model, please do so by simply running main.py. We have simplified the woekflow for python users!

```bash
project-root/
├── main.py
├── GRU                                        # GRU model development 
│   ├── __init__.py  
│   ├── scripts
│   ├── util
│   │   ├── baselines.py
│   │   ├── static_feats.py
│   ├── EVLPMultivariateBreathDataset.py
│   ├── forecast_parameters.py
│   ├── forecast_parameters_w_pred.py
│   ├── forecasting_pipeline.py
│   ├── GRU.py
├── XGB                                       # XGboost model development 
│   ├── __init__.py
│   ├── BaselineModels.py
│   ├── Dataset.py
│   ├── TemporalOrder.py
│   ├── TabularForecasting.py
│   ├── pipelines.py
│   ├── inference.py
│   ├── forecasting_pipeline.py
│   ├── Image PC.py   # update name change
│   ├── Image PC.py   # update name change
│   ├── Image PC.py   # update name change
│   ├── Image PC.py   # update name change
│   ├── utils.py
├── inference
│   ├── __init__.py
│   ├── GRU_inference.py
│   ├── XGB_inference.py
│   ├── XGB_inference_step2.py
│   ├── XGB_PC_dynamic.py
│   ├── XGB_PC_static.py                 # inference run for 
│   ├── data_reforamt.py                 # helper functions for enhanced UI readbility
├── Dockerfile
├── start_docker.sh                      # Running Docker on MacOS/Linux
├── start_docker.bat                     # Running Docker on Windows
├── requirements.txt                     # System requirements
├── GLOSSARY.md                          # for units, ranges, acronyms, and domain terms

```

### Naming convention
This project follows a custom naming convention for model configurations and variable identifiers. Detailed explanations are provided below to enhance clarity and improve code readability.

#### GRU model training setups

A breath setup defines which breaths are included as input and output data to the model. They include:

`A1_A2` Setups for Static digital lung forecasting:

- `A1F50_A2F50`
- `A1F50L50_A2F50`
- `N1L20A1F50L50_A2F50`

`A1PA2_A3` Setups for Static digital lung forecasting:

- `A1F50PA2F50_A3F50`
- `A1F50L50PA2F50_A3F50`
- `N1L20A1F50L50PA2F50_A3F50`

`A1_A3` Setups for Static digital lung forecasting:

- `A1F50_A3F50`
- `A1F50L50_A3F50`
- `N1L20A1F50L50_A3F50`

`A1A2_A3` Setups for Dynamic digital lung forecasting: 

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

## Inference (Create digital twins using your data📊🫁)


## Diagnosing Errors

















