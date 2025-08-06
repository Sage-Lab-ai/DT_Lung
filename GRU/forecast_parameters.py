import torch
import wandb
import argparse
from .forecasting_pipeline import *
from .util.baselines import *
from .GRU import GRU


parser = argparse.ArgumentParser()
parser.add_argument("--setup", type=str, default="A1F50_A2F50", choices=["A1F50_A2F50", "A1F50L50_A2F50", "N1L20A1F50L50_A2F50", "A1F50A2F50_A3F50", "A1F50L50A2F50_A3F50", "N1L20A1F50L50A2F50_A3F50", "A1F50PA2F50_A3F50", "A1F50L50PA2F50_A3F50", "N1L20A1F50L50PA2F50_A3F50", "A1F50_A3F50", "A1F50L50_A3F50", "N1L20A1F50L50_A3F50"])
parser.add_argument("--variable", type=str, default="Dy_comp", choices=["Dy_comp", "P_peak", "P_mean", "Ex_vol"])

parser.add_argument("--gru_hidden_size", type=int, default=100)
parser.add_argument("--num_epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=0.01)
parser.add_argument("--scheduler_step_size", type=int, default=10)
parser.add_argument("--scheduler_gamma", type=float, default=0.5)
parser.add_argument("--criterion", type=str, default="mse_loss")
parser.add_argument("--wandb", action='store_true', default=False)
parser.add_argument("--use_best_hyperparams", action='store_true', default=False)
parser.add_argument("--n_folds", type=int, default=5)
parser.add_argument("--n_seeds", type=int, default=1, choices=[1, 5, 10])
parser.add_argument("--lock_model", action='store_true', default=False)

parser.add_argument("--use_static", action='store_true', default=False)
parser.add_argument("--use_image_pcs", action='store_true', default=False)
parser.add_argument("--normalize_static", action='store_true', default=False)
args = parser.parse_args()



if __name__ == "__main__":
    # Initialize wandb
    if args.wandb:
        wandb.init(
            entity="VentilatorForecasting", 
            project="GRU_DT",
            config=args
        )

    # Load the data excluding the special cases (double to single lungs 
    # on EVLP, short/problematic cases, and cases with tPA applied)
    Dataset_Folder = os.path.join("dataset", "saved_datasets", f"{args.setup}")
    Ys_dict = {args.variable: Ys_files[args.variable]}
    cases_to_exclude = [] # Redacted case numbers
    datasets, selected_cases = get_datasets(Dataset_Folder, Ys_dict, args.setup, cases_to_exclude, args.use_static, args.use_image_pcs)

    param = args.variable
    dataset = datasets[param]

    label = "multivariate"
    if args.use_static and args.use_image_pcs:
        label = "multivariate_w_static_and_img_pcs"

    # ------------------- Baseline forecasting -------------------

    last_30_breath_baseline_performance = last_30_breath_baseline(dataset, param_idx[param])
    entire_input_baseline_performance = entire_input_baseline(dataset, param_idx[param])

    print(f"Last 30 Breath Baseline performance for {param}: {last_30_breath_baseline_performance}")
    print(f"Entire Input Baseline performance for {param}: {entire_input_baseline_performance}")

    if not args.wandb and args.n_folds == 20: 
        # Write the performance metrics to 20_fold_results directory
        write_20_fold_performance(last_30_breath_baseline_performance, args.setup, args.variable, label, clear_first=True)
        write_20_fold_performance(entire_input_baseline_performance, args.setup, args.variable, label, suffix="\n\n")

    # ------------------- GRU Model forecasting -------------------

    # Subtract average of last 30 breaths of X from Y
    avg_of_X = average_of_tail(dataset.X)[:, :, param_idx[param]: param_idx[param] +1]
    dataset.Y = dataset.Y - avg_of_X

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model = GRU

    if not args.wandb and (args.use_best_hyperparams or args.lock_model):
        # Set hyperparameters to the optimal values before calling get_model_training_params(args)
        args = set_best_hyperparams(args, label, args.setup, param)

    model_params, training_params = get_model_training_params(args)

    # Fix the seeds to ensure reproducibility of results
    seeds = [42]
    if args.n_seeds == 5:
        seeds = [42, 83, 123, 456, 789]
    if not args.wandb and args.n_seeds == 10:
        seeds = [42, 83, 123, 456, 789, 54, 108, 987, 654, 321]

    maes = []
    for seed in seeds:
        training_params["seed"] = seed
        
        print(f"Training model for {param} with seed {seed}")

        k = args.n_folds

        if args.lock_model:
            locked_folder = os.path.join("saved_models", args.setup, param, f"seed_{seed}_{label}")
            if not os.path.exists(locked_folder):
                os.makedirs(locked_folder)
            # When training and saving final models, train using all cases, and save resulting model
            train_all_cases(model, model_params, training_params, dataset, device, locked_folder, args.normalize_static)
            continue

        param_folder = os.path.join(Dataset_Folder, param, f"{k}_fold_seed_{seed}_{label}")
        if not os.path.exists(param_folder):
            os.makedirs(param_folder)
        pred_y = train_and_save(model, model_params, training_params, dataset, device, param_folder, args.normalize_static, args.wandb, kfold=k)

        Y_rescaled = dataset.Y + avg_of_X
        pred_y_rescaled = pred_y + avg_of_X
        evaluation_rescaled = evaluate(selected_cases, Y_rescaled, pred_y_rescaled)
            
        print(param, json.dumps(evaluation_rescaled, indent=4))
        if args.wandb:
            maes.append(evaluation_rescaled["mean_mae"])

        if not args.wandb and args.n_folds == 20:
            # Write the results to 20_fold_results directory
            write_20_fold_performance(evaluation_rescaled, args.setup, args.variable, label)

        with open(os.path.join(param_folder, file_evaluation), "w") as f:
            json.dump(evaluation_rescaled, f, indent=4)

        # Short pause between random seeds
        time.sleep(2)

    if args.wandb:
        wandb.log({"mae": np.mean(maes), "gru_hidden_size": args.gru_hidden_size, "learning_rate": args.learning_rate, "scheduler_step_size": args.scheduler_step_size, "scheduler_gamma": args.scheduler_gamma, "criterion": args.criterion})

