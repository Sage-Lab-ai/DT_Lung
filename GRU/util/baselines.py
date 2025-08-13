import numpy as np 

def average_of_tail(X):
    return X[:, -30:].mean(axis=1, keepdims=True)

def average_of_all(X):
    return X.mean(axis=1, keepdims=True)

def get_metrics(true_y, pred_y):
    mean_mae = np.abs(true_y - pred_y).mean(axis=1).mean(axis=0)[0]
    mean_mape = (np.abs((true_y - pred_y) / true_y).mean(axis=1)*100).mean(axis=0)[0]
    std_mae = np.abs(true_y - pred_y).mean(axis=1).std(axis=0)[0]
    std_mape = (np.abs((true_y - pred_y) / true_y).mean(axis=1)*100).std(axis=0)[0]
    return {"mean_mae": mean_mae, "std_mae": std_mae, "mean_mape": mean_mape, "std_mape": std_mape}


def last_30_breath_baseline(dataset, param_index):
    pred_y = average_of_tail(dataset.X)[:, :, param_index:param_index+1]
    return get_metrics(dataset.Y, pred_y)


def entire_input_baseline(dataset, param_index):
    pred_y = average_of_all(dataset.X)[:, :, param_index:param_index+1]
    return get_metrics(dataset.Y, pred_y)
