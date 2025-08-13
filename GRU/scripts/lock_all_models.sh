#!/bin/bash

if [ -z "$1" ]; then
    echo "Please provide the configuration type (multivariate or multivariate_w_static_and_img_pcs)"
    exit 1
fi

declare -a setups=("A1F50_A2F50" "A1F50L50_A2F50" "N1L20A1F50L50_A2F50" "A1F50A2F50_A3F50" "A1F50L50A2F50_A3F50" "N1L20A1F50L50A2F50_A3F50" "A1F50_A3F50" "A1F50L50_A3F50" "N1L20A1F50L50_A3F50")
declare -a variables=("Dy_comp" "P_peak" "P_mean" "Ex_vol")

for setup in "${setups[@]}";
do
    for variable in "${variables[@]}";
    do
        echo "Locking model for setup: $setup and variable: $variable"

        if [ $1 == "multivariate" ]; then
            python forecast_parameters.py --setup $setup --variable $variable --lock_model
        elif [ $1 == "multivariate_w_static_and_img_pcs" ]; then
            python forecast_parameters.py --setup $setup --variable $variable --lock_model --use_static --normalize_static --use_image_pcs
        else
            echo "Invalid configuration type: $1"
            exit 1
        fi

    done
done