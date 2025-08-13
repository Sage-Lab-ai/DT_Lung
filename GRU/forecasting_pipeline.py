import json
import numpy as np
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from .EVLPMultivariateBreathDataset import *
from .util.static_feats import *
import time
from tqdm import tqdm
import wandb
import copy

X_file = "X.npy"
Ys_files = {
    "Dy_comp": "Y_Dy_comp(ml_cmH2O).npy",
    "P_peak": "Y_P_peak(cmH2O).npy",
    "P_mean": "Y_P_mean(cmH2O).npy",
    "Ex_vol": "Y_Ex_vol(ml).npy"
}
file_selected_cases = "cases.json"

param_idx = {
    "Dy_comp": 0,
    "P_peak": 1,
    "P_mean": 2,
    "Ex_vol": 3
}

def get_datasets(Dataset_Folder, Ys_dict, setup, cases_to_exclude, use_static, use_image_pcs):
    # Read X, Y and static data for the setup
    with open(os.path.join(Dataset_Folder, file_selected_cases), "r") as f:
        selected_cases = json.load(f)
    X = np.load(os.path.join(Dataset_Folder, X_file))
    Ys = {param: np.load(os.path.join(Dataset_Folder, file)) for param, file in Ys_dict.items()}
    static = np.load(os.path.join(Dataset_Folder, "static.npy")) if use_static else None

    static_names = StaticFeaturesToInclude[setup]
    static_norms = NormalizationToUse[setup]
    if not use_image_pcs and static is not None:
        static = static[:, :len(static_names)-len(image_pcs)]
        static_norms = static_norms[:len(static_names)-len(image_pcs)]
        print(f"Using static features without image pcs: {static_names[:len(static_names)-len(image_pcs)]}, {static.shape}")
    elif static is not None:
        print(f"Using all static features including image pcs: {static_names}, {static.shape}")
    
    # Remove cases in cases_to_exclude
    indices_to_drop = [selected_cases.index(case) for case in cases_to_exclude if case in selected_cases]
    selected_cases = [case for case in selected_cases if case not in cases_to_exclude]

    X = np.delete(X, indices_to_drop, axis=0)
    print(f"X Shape : {X.shape}")

    for param, Y in Ys.items():
        Ys[param] = np.delete(Y, indices_to_drop, axis=0)
        print(f"Y Shape {param} : {Ys[param].shape}")
    
    if static is not None:
        static = np.delete(static, indices_to_drop, axis=0)
        print(f"Static Shape : {static.shape}")

    # Create the EVLPMultivariateBreathDataset for each parameter
    datasets = {param: EVLPMultivariateBreathDataset(selected_cases, X, Y, static, static_norm=static_norms) for param, Y in Ys.items()}

    return datasets, selected_cases


def write_20_fold_performance(performance, setup, variable, label, clear_first=False, suffix="\n"):
    # Folder for writing 20-fold cv results
    TwentyFold_Folder = os.path.join("20_fold_results", f"{setup}", f"{variable}")
    if not os.path.exists(TwentyFold_Folder):
        os.makedirs(TwentyFold_Folder)
    
    for metric in ["mean_mae", "std_mae", "mean_mape", "std_mape"]:
        with open(os.path.join(TwentyFold_Folder, f"results_{setup}_{variable}_{label}_{metric}.txt"), "w" if clear_first else "a") as f:
            f.write(f"{performance[metric]}{suffix}")


def set_best_hyperparams(args, label, setup, variable):
    # Read the sweep_id for the given setup and variable
    sweep_id_file = os.path.join("hyperparameter_sweeps", f"gru_{label}_sweep_ids.json")
    with open(sweep_id_file, "r") as f:
        sweep_ids = json.load(f)
        if setup not in sweep_ids or variable not in sweep_ids[setup]:
            raise ValueError("Invalid setup or variable")
        sweep_id = sweep_ids[setup][variable]

    # Set the best hyperparameters from the sweep
    with open(f"hyperparameter_sweeps/best_hyperparameters/{sweep_id}_best_hyperparams.json", "r") as f:
        best_hyperparams = json.load(f)
        args.gru_hidden_size = best_hyperparams["gru_hidden_size"]
        args.num_epochs = best_hyperparams["num_epochs"]
        args.batch_size = best_hyperparams["batch_size"]
        args.learning_rate = best_hyperparams["learning_rate"]
        args.scheduler_step_size = best_hyperparams["scheduler_step_size"]
        args.scheduler_gamma = best_hyperparams["scheduler_gamma"]
        args.criterion = best_hyperparams["criterion"]
        print("Using best hyperparameters from sweep")
    
    return args


def get_model_training_params(args):
    # model parameters
    model_params = {
        "gru_hidden_size": args.gru_hidden_size,
        "mlp_hidden_size": [args.gru_hidden_size + 20, args.gru_hidden_size],
        "mlp_activation": "ReLU",
        "GRU_params": {"num_layers": 1, "bidirectional": False}
    }

    # Training parameters
    training_params = {
        "criterion": args.criterion,
        "optimizer_fn": "Adam",
        "optimizer_params": {"lr": args.learning_rate},
        "scheduler": "StepLR",
        "scheduler_params": {"step_size": args.scheduler_step_size, "gamma": args.scheduler_gamma},
        "num_epochs": args.num_epochs,
        "print_interval": None,
        "batch_size": args.batch_size,
        "seed": 42
    }

    return model_params, training_params


# Files to save the results
file_training_losses = "training_losses.npy"
file_validation_losses = "validation_losses.npy"
file_pred_Y = "pred_Y.npy"
file_pred_Y_rescaled = "pred_Y_rescaled.npy"
file_true_Y = "true_Y.npy"
file_true_Y_rescaled = "true_Y_rescaled.npy"
file_evaluation = "evaluation.json"


def train_and_save(model, model_params, training_params, dataset, device, save_folder, normalize_static, using_wandb, kfold=-1):
    create_model_fn = lambda: model(num_dynamic_feature=dataset.X.shape[-1],
                                    output_sequence_length=dataset.Y.shape[1],
                                    num_static_feature=dataset.static.shape[-1] if dataset.static is not None else 0,
                                    **model_params)
    training_loss_traces, validation_loss_traces, pred_y = nn_kfold_cv(dataset, device, dataset.selected_cases, create_model_fn,
                                                                       training_params, normalize_static, using_wandb, kfold)
    np.save(os.path.join(save_folder, file_training_losses), training_loss_traces)
    np.save(os.path.join(save_folder, file_validation_losses), validation_loss_traces)

    np.save(os.path.join(save_folder, file_pred_Y), pred_y)
    np.save(os.path.join(save_folder, file_true_Y), dataset.Y)
    return pred_y


def train_and_save_w_separate_train_and_inference_datasets(model, model_params, training_params, dataset_for_training, dataset_for_inference, device, save_folder, use_static, normalize_static, kfold=-1, true_y_from_inference=False):
    create_model_fn = lambda: model(num_dynamic_feature=dataset_for_training.X.shape[-1],
                                    output_sequence_length=dataset_for_training.Y.shape[1],
                                    num_static_feature=dataset_for_training.static.shape[-1] if dataset_for_training.static is not None else 0,
                                    **model_params)
    training_loss_traces, validation_loss_traces, pred_y, true_y = nn_kfold_cv_w_separate_train_and_inference_datasets(dataset_for_training, dataset_for_inference, device, dataset_for_training.selected_cases, dataset_for_inference.selected_cases, create_model_fn,
                                                                       training_params, use_static, normalize_static, kfold, true_y_from_inference)
    np.save(os.path.join(save_folder, file_training_losses), training_loss_traces)
    np.save(os.path.join(save_folder, file_validation_losses), validation_loss_traces)

    np.save(os.path.join(save_folder, file_pred_Y), pred_y)
    np.save(os.path.join(save_folder, file_true_Y), true_y)
    return pred_y, true_y


def train_all_cases(model, model_params, training_params, dataset_orig, device, save_folder, normalize_static):    
    dataset = copy.deepcopy(dataset_orig) # Make a copy so that the static features in original dataset are not modified
    if dataset.static is not None and normalize_static:
        # normalize static features
        static_mean = dataset.static.mean(axis=0)
        static_std = dataset.static.std(axis=0)
        dataset.static = (dataset.static - static_mean) / static_std

    model = model(num_dynamic_feature=dataset.X.shape[-1],
                                    output_sequence_length=dataset.Y.shape[1],
                                    num_static_feature=dataset.static.shape[-1] if dataset.static is not None else 0,
                                    **model_params).to(device)

    training_losses, _, _ = nn_train(model=model, device=device, train_data=dataset, fold=None, using_wandb=False, val_data=None, **training_params)
    print("Training losses:", training_losses)

    # Save the model and training losses 
    torch.save(model, os.path.join(save_folder, f"locked_model.pt"))
    np.save(os.path.join(save_folder, file_training_losses), training_losses)


def timeit(func):
    """
    Decorator to time the function
    """

    def timed(*args, **kwargs):
        print(f"Function <{func.__name__}> started")
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Function <{func.__name__}> took: {round((end - start), 2)} s")
        return result

    return timed


@timeit
def nn_kfold_cv(dataset_orig, device, selected_cases, create_model_fn, train_model_params, normalize_static, using_wandb, k=-1):
    num_samples = len(dataset_orig)
    training_loss_traces = []
    validation_loss_traces = []
    pred_Y = np.zeros_like(dataset_orig.Y)
    if k == -1:
        k = num_samples

    for i in range(k):
        dataset = copy.deepcopy(dataset_orig) # Make a copy so that the static features in original dataset are not modified
        val_indices = [c for c in range(i, num_samples, k)]
        train_indices = list(set(range(num_samples)) - set(val_indices))
        print(f"Fold {i + 1}/{k} : EVLP {', '.join([str(selected_cases[ind]) for ind in val_indices])}")

        train_subset = Subset(dataset, train_indices)
        val_subset = Subset(dataset, val_indices)

        if dataset.static is not None and normalize_static:
            # normalize static features
            static_mean = dataset.static[train_indices].mean(axis=0)
            static_std = dataset.static[train_indices].std(axis=0) + 1e-6 # Add a small value to avoid division by zero
            # only normalize the "Z-score" static features
            cols_to_normalize = [i for i, norm in enumerate(dataset.static_norm) if norm == "Z-score"]
            dataset.static[:, cols_to_normalize] = (dataset.static[:, cols_to_normalize] - static_mean[cols_to_normalize]) / static_std[cols_to_normalize]

        model = create_model_fn().to(device)

        training_losses, validation_losses, y_pred = nn_train(model=model, device=device, train_data=train_subset, fold=i+1, using_wandb=using_wandb,
                                                              val_data=val_subset, **train_model_params)

        training_loss_traces.append(training_losses)
        validation_loss_traces.append(validation_losses)
        pred_Y[val_indices] = y_pred

        y_true = np.array(dataset[val_indices][1])
        mae = np.mean(np.abs(y_pred - y_true), axis=1)
        print(f"MAE: {mae}\n")

    return np.array(training_loss_traces), np.array(validation_loss_traces), pred_Y


@timeit
def nn_kfold_cv_w_separate_train_and_inference_datasets(dataset_for_training_orig, dataset_for_inference_orig, device, selected_cases_training, selected_cases_inference, create_model_fn, train_model_params, use_static, normalize_static, k=-1, true_y_from_inference=False):
    num_samples = len(dataset_for_inference_orig)
    training_loss_traces = []
    validation_loss_traces = []
    pred_Y = np.zeros_like(dataset_for_inference_orig.Y)
    true_Y = np.zeros_like(dataset_for_inference_orig.Y)
    if k == -1:
        k = num_samples

    for i in range(k): 
        dataset_for_training = copy.deepcopy(dataset_for_training_orig) # Make a copy so that the static features in original dataset are not modified
        dataset_for_inference = copy.deepcopy(dataset_for_inference_orig) # Make a copy so that the static features in original dataset are not modified
        val_indices_dataset_inference = [c for c in range(i, num_samples, k)]
        cases_of_val_dataset = [selected_cases_inference[c] for c in val_indices_dataset_inference]

        print(f"Fold {i + 1}/{k} : EVLP {', '.join([str(selected_cases_inference[c]) for c in val_indices_dataset_inference])}")

        train_indices_dataset_training = [c for c in range(len(selected_cases_training)) if selected_cases_training[c] not in cases_of_val_dataset]
        val_indices_dataset_training = [selected_cases_training.index(c) for c in cases_of_val_dataset]
        # assert that intersection of train_indices_dataset and val_indices_dataset is empty
        assert len(set(train_indices_dataset_training).intersection(set(val_indices_dataset_training))) == 0
        assert len(set(train_indices_dataset_training).union(set(val_indices_dataset_training))) == len(selected_cases_training)

        train_subset = Subset(dataset_for_training, train_indices_dataset_training)
        val_subset = Subset(dataset_for_inference, val_indices_dataset_inference)

        if use_static and normalize_static:
            # normalize static features
            static_mean = dataset_for_training.static[train_indices_dataset_training].mean(axis=0)
            static_std = dataset_for_training.static[train_indices_dataset_training].std(axis=0) + 1e-6 # Add a small value to avoid division by zero
            # only normalize the "Z-score" static features
            cols_to_normalize = [i for i, norm in enumerate(dataset_for_training.static_norm) if norm == "Z-score"]
            dataset_for_training.static[:, cols_to_normalize] = (dataset_for_training.static[:, cols_to_normalize] - static_mean[cols_to_normalize]) / static_std[cols_to_normalize]
            dataset_for_inference.static[:, cols_to_normalize] = (dataset_for_inference.static[:, cols_to_normalize] - static_mean[cols_to_normalize]) / static_std[cols_to_normalize]

        model = create_model_fn().to(device)

        training_losses, validation_losses, y_pred = nn_train(model=model, device=device, train_data=train_subset, 
                                                              fold=i+1, using_wandb=False,
                                                              val_data=val_subset, **train_model_params)

        training_loss_traces.append(training_losses)
        validation_loss_traces.append(validation_losses)
        pred_Y[val_indices_dataset_inference] = y_pred

        if true_y_from_inference:
            y_true = np.array(dataset_for_inference[val_indices_dataset_inference][1])
            # because dataset_for_training is A1A2_A3, but dataset_for_inference is A1PA2_A3, and here, 
            # A3 from A1A2_A3 has been rescaled by the average of the tail of A2, whereas A3 from A1PA2_A3
            # has been rescaled by the average of the tail of PA2.
        else:
            y_true = np.array(dataset_for_training[val_indices_dataset_training][1]) 
            # because dataset_for_training is A1_A2, but dataset_for_inference is A1_A3
        true_Y[val_indices_dataset_inference] = y_true
        mae = np.mean(np.abs(y_pred - y_true), axis=1)
        print(f"MAE: {mae}")
        print()

    return np.array(training_loss_traces), np.array(validation_loss_traces), pred_Y, true_Y


def nn_train(model, device, train_data, criterion, optimizer_fn, optimizer_params, fold, using_wandb, val_data=None,
             scheduler=None, scheduler_params=None, num_epochs=100, print_interval=None, batch_size=32, seed=42):
    torch.manual_seed(seed)
    model.to(device)

    criterion = getattr(F, criterion)

    using_cuda = device.type == "cuda"

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=using_cuda)
    optimizer_fn = getattr(torch.optim, optimizer_fn)
    optimizer = optimizer_fn(model.parameters(), **optimizer_params)

    if scheduler is not None:
        scheduler = getattr(torch.optim.lr_scheduler, scheduler)
        scheduler = scheduler(optimizer, **scheduler_params)

    if val_data is not None:
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=using_cuda)
    else:
        val_loader = None

    training_losses, validation_losses = [], []

    for epoch in tqdm(range(num_epochs), desc="Training", unit="epoch"):
        model.train()
        training_loss = 0
        validation_loss = 0

        for (x, s), y in train_loader:
            x = x.to(device, non_blocking=True)
            s = s.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            y_pred = model(x, s)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

        training_losses.append(training_loss / len(train_loader))

        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                for (x, s), y in val_loader:
                    x = x.to(device, non_blocking=True)
                    s = s.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    y_pred = model(x, s)
                    loss = criterion(y_pred, y)
                    validation_loss += loss.item()
                validation_losses.append(validation_loss / len(val_loader))

        if using_wandb:
            if val_loader is not None:
                wandb.log({f"training_loss_fold_{fold}_seed_{seed}": training_losses[-1], f"validation_loss_fold_{fold}_seed_{seed}": validation_losses[-1]})
            else:
                wandb.log({f"training_loss_fold_{fold}_seed_{seed}": training_losses[-1]})

        if scheduler is not None:
            scheduler.step()

        if print_interval is not None and (epoch + 1) % print_interval == 0:
            msg = f"Epoch [{epoch + 1}/{num_epochs}] : Training Loss: {training_losses[-1]:.4f}"
            if val_data is not None:
                msg += f" | Validation Loss: {validation_losses[-1]:.4f}"
            print(msg)

    if val_loader is not None:
        model.eval()
        y_preds = []
        with torch.no_grad():
            for (x, s), y in val_loader:
                x = x.to(device, non_blocking=True)
                s = s.to(device, non_blocking=True)
                y_pred = model(x, s)
                y_preds.append(y_pred.detach().cpu().numpy())
        predicted_val_y = np.concatenate(y_preds)
    else:
        predicted_val_y = None

    return training_losses, validation_losses, predicted_val_y


def evaluate(selected_cases, true_y, pred_y, k=5):
    maes = np.array([(np.mean(np.abs(true_y[i] - pred_y[i])), selected_cases[i]) for i in range(len(selected_cases))])
    mapes = np.array([(np.mean(np.abs((true_y[i] - pred_y[i]) / true_y[i]))*100, selected_cases[i]) for i in range(len(selected_cases))])

    k = min(k, maes.shape[0])

    average_mae = np.mean(maes[:, 0])
    std_mae = np.std(maes[:, 0])

    average_mape = np.mean(mapes[:, 0])
    std_mape = np.std(mapes[:, 0])

    output = {}

    output["mean_mae"] = average_mae
    output["std_mae"] = std_mae

    output["mean_mape"] = average_mape
    output["std_mape"] = std_mape

    worst_cases = maes[:, 0].argsort()[-k:][::-1]
    output[f"mean mae without worst {k} cases"] = np.mean(np.delete(maes[:, 0], worst_cases))
    output[f"worst {k} cases"] = {}
    for case in worst_cases:
        output[f"worst {k} cases"][int(maes[case, 1])] = maes[case, 0]

    return output
