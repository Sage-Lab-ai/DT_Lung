#!/bin/bash

if [ -z "$1" ]; then
    echo "Please provide the configuration type (multivariate or multivariate_w_static_and_img_pcs)"
    exit 1
fi

declare -a setups=("A1F50_A2F50" "A1F50L50_A2F50" "N1L20A1F50L50_A2F50" "A1F50A2F50_A3F50" "A1F50L50A2F50_A3F50" "N1L20A1F50L50A2F50_A3F50" "A1F50PA2F50_A3F50" "A1F50L50PA2F50_A3F50" "N1L20A1F50L50PA2F50_A3F50" "A1F50_A3F50" "A1F50L50_A3F50" "N1L20A1F50L50_A3F50")
declare -a variables=("Dy_comp" "P_peak" "P_mean" "Ex_vol")
declare -a pred_setups=("A1F50PA2F50_A3F50" "A1F50L50PA2F50_A3F50" "N1L20A1F50L50PA2F50_A3F50")

for setup in "${setups[@]}";
do
    for variable in "${variables[@]}";
    do
        echo "Running 20-fold CV experiments for setup: $setup and variable: $variable"

        if [[ " ${pred_setups[@]} " =~ " ${setup} " ]]; then
            forecasting_file="forecast_parameters_w_pred.py"
        else
            forecasting_file="forecast_parameters.py"
        fi

        if [ $1 == "multivariate" ]; then
            # Note: For a given setup, only redo A2 predictions for all 4 parameters for the first variable of A3 forecasting (as these predictions
            # can be reused) by using the --overwrite_preds flag.
            if [[ "$variable" == "${variables[0]}" && " ${pred_setups[@]} " =~ " ${setup} " ]]; then
                python $forecasting_file --setup $setup --variable $variable --use_best_hyperparams --n_folds 20 --n_seeds 10 --overwrite_preds
            else
                python $forecasting_file --setup $setup --variable $variable --use_best_hyperparams --n_folds 20 --n_seeds 10
            fi
        elif [ $1 == "multivariate_w_static_and_img_pcs" ]; then
            if [[ "$variable" == "${variables[0]}" && " ${pred_setups[@]} " =~ " ${setup} " ]]; then
                python $forecasting_file --setup $setup --variable $variable --use_best_hyperparams --n_folds 20 --n_seeds 10 --use_static --normalize_static --use_image_pcs --overwrite_preds
            else
                python $forecasting_file --setup $setup --variable $variable --use_best_hyperparams --n_folds 20 --n_seeds 10 --use_static --normalize_static --use_image_pcs
            fi
        else
            echo "Invalid configuration type: $1"
            exit 1
        fi

    done
done