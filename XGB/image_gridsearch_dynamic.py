import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import os
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import shap
import pickle

save_model_dir = '../Model/XGB_PC/models_dynamic'
save_dir = '../Model/XGB_PC/results_dynamic_hyper'
os.makedirs(save_dir, exist_ok=True)

# Data Merging
tab_data_df = pd.read_csv('df_hourly 1h2h.csv', index_col='Id No')
image_pc1h_path_torex = 'pca-1h-features.csv'
image_pc3h_path_torex = 'pca-3h-features.csv'
image_pc1h_torex = pd.read_csv(image_pc1h_path_torex, index_col='Id No')
image_pc3h_torex = pd.read_csv(image_pc3h_path_torex, index_col='Id No')

image_pc1h_path = 'pca-1h-features.csv'
image_pc3h_path = 'pca-3h-features.csv'
image_pc1h = pd.read_csv(image_pc1h_path, index_col='Id No')
image_pc3h = pd.read_csv(image_pc3h_path, index_col='Id No')

image_pc1h = pd.concat([image_pc1h, image_pc1h_torex])
image_pc3h = pd.concat([image_pc3h, image_pc3h_torex])
image_pc1h.columns = ['1h_' + col_name for col_name in image_pc1h.columns if col_name != 'Id No']
image_pc3h.columns = ['3h_' + col_name for col_name in image_pc3h.columns if col_name != 'Id No']

merged_df = pd.merge(tab_data_df, image_pc1h, on='Id No')
merged_df = pd.merge(merged_df, image_pc3h, on='Id No')

# log trans
columns_to_transform = [col for col in merged_df.columns if 'STEEN' in col or 'PVR' in col]
merged_df[columns_to_transform] = merged_df[columns_to_transform].apply(np.log1p)
merged_df = merged_df.dropna()
merged_df = merged_df.sample(frac=1.0, random_state=42)
y = merged_df[[col_name for col_name in merged_df.columns if '3h_' in col_name]]
X= merged_df[[col_name for col_name in merged_df.columns if col_name not in y.columns]]
X= X[[col_name for col_name in X.columns if not 'STEEN' in col_name]]
results = []
n_jobs = 10

to_tune = [{
    'model__objective': ["reg:squarederror", "reg:absoluteerror"],
    'model__n_estimators': [...], 
    'model__learning_rate': [...],
    'model__gamma': [...],
    'model__max_depth': [...],
    'model__min_child_weight': [...],
    'model__alpha': [...], 
    'model__lambda': [...]
}]

select_keys = list(to_tune[0].keys())

scores = ['neg_mean_absolute_error' , 'neg_mean_squared_error'] #'r2'
categorical_list = [...]
numeric_list = [x for x in X.columns if x not in categorical_list]


for col_name in y.columns: 
    np.random.seed(42) 
    y_col = y[col_name]
    
    pipeline = XGBRegressor(n_jobs=n_jobs)
    pipeline.fit(X, y_col)
    explainer = shap.Explainer(pipeline, X)

    shap_values = explainer(X)
    mean_shap_values = np.abs(shap_values.values).mean(axis=0)

    if col_name == '3h_feature_pca_0':
        N = 5
    else:
        N = 5
    top_indices = np.argsort(mean_shap_values)[-N:]
    top_features = X.columns[top_indices]
    pca = col_name.replace('3h_feature_', '')
    with open(os.path.join(save_dir, f'top_features_{pca}.txt'), 'w') as f:
        for feature in top_features:
            f.write(f"{feature}\n")
    Xfeat = X[top_features]
    Xfeat = X[top_features]

    pipeline = Pipeline([
        ('scaler', StandardScaler()),  
        ('model', XGBRegressor(n_jobs=n_jobs))  
    ])

    grid_search_model = GridSearchCV(pipeline, param_grid=to_tune, 
                                     scoring='neg_mean_squared_error', return_train_score=True,
                                     cv=KFold(n_splits=10)) 
    grid_search_model.fit(Xfeat, y_col)
    best_pipeline = grid_search_model.best_estimator_
    with open(os.path.join(save_model_dir, f'{col_name}_model.pkl'), 'wb') as file:
        pickle.dump(best_pipeline, file)
    
    results_grid = grid_search_model.cv_results_

    best_params = grid_search_model.best_params_
    best_index = None
    for i, params in enumerate(results_grid['params']):
        if params == best_params:
            best_index = i
            break

    for cv, (tr_indices, ts_indices) in enumerate(grid_search_model.cv.split(X)):
        tr_x, ts_x = Xfeat.iloc[tr_indices], Xfeat.iloc[ts_indices]
        tr_y, ts_y = y.iloc[tr_indices], y.iloc[ts_indices]
        val_loss = -results_grid[f'split{cv}_test_score'][best_index]
        tr_loss = -results_grid[f'split{cv}_train_score'][best_index]
        baseline_ts_x = X.iloc[ts_indices]
        baseline_mse = mean_squared_error(baseline_ts_x[col_name.replace('3h', '1h')], ts_y[col_name])

    y_pred = grid_search_model.predict(Xfeat)
    y_col_test = y[col_name]
    mse = mean_squared_error(y_col_test, y_pred)
    mae = mean_absolute_error(y_col_test, y_pred)
    mpe = np.mean(np.abs((y_col_test - y_pred) / y_col_test)) * 100
    best_estimator_params = grid_search_model.best_estimator_.get_params()
    best_estimator_params = {key: best_estimator_params[key] for key in select_keys if key in best_estimator_params}
    params = {'column_name': col_name, 'mse': mse, 'mae': mae, 'mpe': mpe}
    params.update(best_estimator_params)
    results.append(params)

results_df = pd.DataFrame(results)
results_df.to_csv(save_dir + '/results.csv', index=False)

