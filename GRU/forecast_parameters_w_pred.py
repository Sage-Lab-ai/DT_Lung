import torch
import argparse
from .forecasting_pipeline import *
from .util.baselines import *
from .GRU import GRU

parser = argparse.ArgumentParser()
parser.add_argument("--setup", type=str, default="A1F50PA2F50_A3F50", choices=["A1F50PA2F50_A3F50", "A1F50L50PA2F50_A3F50", "N1L20A1F50L50PA2F50_A3F50"])
parser.add_argument("--variable", type=str, default="Dy_comp", choices=["Dy_comp", "P_peak", "P_mean", "Ex_vol"])

parser.add_argument("--gru_hidden_size", type=int, default=100)
parser.add_argument("--num_epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=0.01)
parser.add_argument("--scheduler_step_size", type=int, default=10)
parser.add_argument("--scheduler_gamma", type=float, default=0.5)
parser.add_argument("--criterion", type=str, default="mse_loss")

parser.add_argument("--use_best_hyperparams", action='store_true', default=False)
parser.add_argument("--n_folds", type=int, default=5)
parser.add_argument("--n_seeds", type=int, default=1, choices=[1, 5, 10])

parser.add_argument("--use_static", action='store_true', default=False)
parser.add_argument("--use_image_pcs", action='store_true', default=False)
parser.add_argument("--normalize_static", action='store_true', default=False)

parser.add_argument("--overwrite_preds", action='store_true', default=False)
args = parser.parse_args()



if __name__ == "__main__":

    static_twin_datasets = {
        "A1F50PA2F50_A3F50": ["A1F50_A3F50", "A1F50_A2F50", "A1F50A2F50_A3F50"], 
        "A1F50L50PA2F50_A3F50": ["A1F50L50_A3F50", "A1F50L50_A2F50", "A1F50L50A2F50_A3F50"], 
        "N1L20A1F50L50PA2F50_A3F50": ["N1L20A1F50L50_A3F50", "N1L20A1F50L50_A2F50", "N1L20A1F50L50A2F50_A3F50"]
    } 
    
    Dataset_Folder = os.path.join("dataset", "saved_datasets", f"{args.setup}")
    A1_A3_Dataset_Folder = os.path.join("dataset", "saved_datasets", static_twin_datasets[args.setup][0])
    A1_A2_Dataset_Folder = os.path.join("dataset", "saved_datasets", static_twin_datasets[args.setup][1])
    A1A2_A3_Dataset_Folder = os.path.join("dataset", "saved_datasets", static_twin_datasets[args.setup][2])

    cases_to_exclude = [] # Redacted case numbers

    datasets_a1_a3, selected_cases_a1_a3 = get_datasets(A1_A3_Dataset_Folder, Ys_files, static_twin_datasets[args.setup][0], cases_to_exclude, args.use_static, args.use_image_pcs)
    datasets_a1_a2, selected_cases_a1_a2 = get_datasets(A1_A2_Dataset_Folder, Ys_files, static_twin_datasets[args.setup][1], cases_to_exclude, args.use_static, args.use_image_pcs)
    datasets_a1a2_a3, selected_cases_a1a2_a3 = get_datasets(A1A2_A3_Dataset_Folder, Ys_files, static_twin_datasets[args.setup][2], cases_to_exclude, False, False)

    Ys_a1_a3 = {param: dataset.Y.copy() for param, dataset in datasets_a1_a3.items()}

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model = GRU
    
    label = "multivariate"
    if args.use_static and args.use_image_pcs:
        label = "multivariate_w_static_and_img_pcs"
        
    # ------------------- Baseline forecasting -------------------
    for (param_a1_a3, dataset_a1_a3) in datasets_a1_a3.items():
        last_30_breath_baseline_performance = last_30_breath_baseline(dataset_a1_a3, param_idx[param_a1_a3])
        entire_input_baseline_performance = entire_input_baseline(dataset_a1_a3, param_idx[param_a1_a3])
        print(f"Last 30 Breath Baseline performance for {param_a1_a3}: {last_30_breath_baseline_performance}")
        print(f"Entire Input Baseline performance for {param_a1_a3}: {entire_input_baseline_performance}")
        
        if args.n_folds == 20 and param_a1_a3 == args.variable:
            write_20_fold_performance(last_30_breath_baseline_performance, args.setup, args.variable, label, clear_first=True)
            write_20_fold_performance(entire_input_baseline_performance, args.setup, args.variable, label, suffix="\n\n")


    # ------------------- GRU Model forecasting -------------------
    
    # Subtract average of the last 30 breaths of X from Y for each dataset type
    avg_of_tail_X_a1_a3 = average_of_tail(datasets_a1_a3[args.variable].X)
    avg_of_tail_X_a1_a2 = average_of_tail(datasets_a1_a2[args.variable].X)
    avg_of_tail_X_a1a2_a3 = average_of_tail(datasets_a1a2_a3[args.variable].X)

    for (param_a1_a3, dataset_a1_a3), (_, dataset_a1_a2), (_, dataset_a1a2_a3) in zip(datasets_a1_a3.items(), datasets_a1_a2.items(), datasets_a1a2_a3.items()):
        dataset_a1_a3.Y = dataset_a1_a3.Y - avg_of_tail_X_a1_a3[:, :, param_idx[param_a1_a3]:param_idx[param_a1_a3]+1] 
        dataset_a1_a2.Y = dataset_a1_a2.Y - avg_of_tail_X_a1_a2[:, :, param_idx[param_a1_a3]:param_idx[param_a1_a3]+1]
        dataset_a1a2_a3.Y = dataset_a1a2_a3.Y - avg_of_tail_X_a1a2_a3[:, :, param_idx[param_a1_a3]:param_idx[param_a1_a3]+1]

    seeds = [42]
    if args.n_seeds == 5:
        seeds = [42, 83, 123, 456, 789]
    if args.n_seeds == 10:
        seeds = [42, 83, 123, 456, 789, 54, 108, 987, 654, 321]
     
    for seed in seeds:
        # First, do 20-fold predictions for A1_dataset (from A1_A3) where for each fold,
        # train using all cases in A1_A2 not in current validation set of fold. 
        # Save predictions as predA2
        setup = static_twin_datasets[args.setup][1]

        for (param_a1_a3, dataset_a1_a3), (_, dataset_a1_a2) in zip(datasets_a1_a3.items(), datasets_a1_a2.items()):
            print(f"Training model for {param_a1_a3} {setup}") 
            k = args.n_folds

            args = set_best_hyperparams(args, label, setup, param_a1_a3)
            model_params, training_params = get_model_training_params(args)
            training_params["seed"] = seed

            # Check if training was already done for this setup, variable, seed, and label and if so, skip the training.
            # This is so that for each value of args.variable being forecasted for A3 (with same setup, seed, and label), the
            # same predictions are used for all params for A2, but you can force the training to be done again by setting
            # args.overwrite_preds to True.
            param_folder = os.path.join(Dataset_Folder, setup, param_a1_a3, f"{k}_fold_seed_{seed}_{label}")
            # Note: the param_folder directory is in the format dataset/saved_datasets/A1PA2_A3_setup/A1_A2_setup/param/k_fold_seed_seed_label
            if os.path.exists(param_folder) and not args.overwrite_preds:
                print("Skipping training for", param_a1_a3, setup, "as it is already done.")
                continue
            if not os.path.exists(param_folder):
                os.makedirs(param_folder)

            pred_y_a2, true_y = train_and_save_w_separate_train_and_inference_datasets(model, model_params, training_params, dataset_a1_a2, dataset_a1_a3, device, param_folder, args.use_static, args.normalize_static, kfold=k)
            
            true_y_rescaled = true_y + avg_of_tail_X_a1_a3[:, :, param_idx[param_a1_a3]:param_idx[param_a1_a3]+1]
            pred_y_a2_rescaled = pred_y_a2 + avg_of_tail_X_a1_a3[:, :, param_idx[param_a1_a3]:param_idx[param_a1_a3]+1]
            np.save(os.path.join(param_folder, file_pred_Y_rescaled), pred_y_a2_rescaled)
            evaluation_rescaled = evaluate(selected_cases_a1_a3, true_y_rescaled, pred_y_a2_rescaled)
            print(param_a1_a3, setup, json.dumps(evaluation_rescaled, indent=4))

            time.sleep(2)

        # Combine A1 (from A1_A3) with pred_y_a2_rescaled to get A1predA2 dataset
        selected_cases_a1preda2_a3 = selected_cases_a1_a3
        all_rescaled_pred_ys = []
        for p in Ys_a1_a3.keys():
            param_folder = os.path.join(Dataset_Folder, setup, p, f"{k}_fold_seed_{seed}_{label}")
            pred_y_rescaled = np.load(os.path.join(param_folder, file_pred_Y_rescaled))
            all_rescaled_pred_ys.append(pred_y_rescaled)
        all_rescaled_pred_ys = np.concatenate(all_rescaled_pred_ys, axis=2)

        X_a1preda2_a3 = np.concatenate([datasets_a1_a3[args.variable].X, all_rescaled_pred_ys], axis=1)
        Ys_a1preda2_a3 = Ys_a1_a3
        static_a1preda2_a3 = None

        datasets_a1preda2_a3 = {param: EVLPMultivariateBreathDataset(selected_cases_a1preda2_a3, X_a1preda2_a3, Y, static_a1preda2_a3) for param, Y in Ys_a1preda2_a3.items()}

        # Second, do 20-fold preidctions for A1predA2 dataset where for each fold,
        # train the model using all cases in A1A2_A3 dataset not in current 
        # validation set of fold.

        # Start by subtracting the average of the last 30 breaths of X from Y for each dataset type
        avg_of_tail_X_a1preda2_a3 = average_of_tail(datasets_a1preda2_a3[args.variable].X) 
        for param_a1preda2_a3, dataset_a1preda2_a3 in datasets_a1preda2_a3.items():
            dataset_a1preda2_a3.Y = dataset_a1preda2_a3.Y - avg_of_tail_X_a1preda2_a3[:, :, param_idx[param_a1preda2_a3]:param_idx[param_a1preda2_a3]+1]
 
        setup = static_twin_datasets[args.setup][2]
        args = set_best_hyperparams(args, label, setup, args.variable)
        model_params, training_params = get_model_training_params(args)
        training_params["seed"] = seed

        param_a1preda2_a3 = args.variable
        dataset_a1preda2_a3 = datasets_a1preda2_a3[param_a1preda2_a3]
        dataset_a1a2_a3 = datasets_a1a2_a3[param_a1preda2_a3]

        print(f"Training model for {param_a1preda2_a3} {setup}")
        k = args.n_folds

        param_folder = os.path.join(Dataset_Folder, setup, param_a1preda2_a3, f"{k}_fold_seed_{seed}_{label}")
        if not os.path.exists(param_folder):
            os.makedirs(param_folder)

        # Note: Cannot use static features in this train and inference step because that would be allowing model to see real information about 2nd hour or A2_A3 static features of case
        # (and these are static models, which should only see 1st hour and A1_A2 static information)
        pred_y_a3, true_y = train_and_save_w_separate_train_and_inference_datasets(model, model_params, training_params, dataset_a1a2_a3, dataset_a1preda2_a3, device, param_folder, False, False, kfold=k, true_y_from_inference=True)

        # 7. Compare these predictions with true A3 predictions            
        true_y_rescaled = true_y + avg_of_tail_X_a1preda2_a3[:, :, param_idx[param_a1preda2_a3]:param_idx[param_a1preda2_a3]+1]
        pred_y_a3_rescaled = pred_y_a3 + avg_of_tail_X_a1preda2_a3[:, :, param_idx[param_a1preda2_a3]:param_idx[param_a1preda2_a3]+1]
        np.save(os.path.join(param_folder, file_pred_Y_rescaled), pred_y_a3_rescaled)
        np.save(os.path.join(param_folder, file_true_Y_rescaled), true_y_rescaled)
        evaluation_rescaled = evaluate(selected_cases_a1preda2_a3, true_y_rescaled, pred_y_a3_rescaled)
        print(param_a1preda2_a3, setup, json.dumps(evaluation_rescaled, indent=4))

        if args.n_folds == 20:
            write_20_fold_performance(evaluation_rescaled, args.setup, args.variable, label)
        
        time.sleep(2)