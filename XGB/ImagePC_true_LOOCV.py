
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import datetime
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import os
feat_dir = '../Model/XGB_PC/results_dynamic_hyper'
save_dir = '../Model/XGB_PC/results_dynamic_LOOCV'
os.makedirs(save_dir, exist_ok=True)

# Data Merging
tab_data_df = pd.read_csv('df_hourly1h2h.csv', index_col='Id No')
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

#%%

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

selected_hypers = pd.read_csv('../Model/XGB_PC/results_dynamic_hyper/results.csv')

to_tune = [{
    'model__objective': [],
    'model__n_estimators': [],
    'model__learning_rate': [],
    'model__gamma': [],
    'model__max_depth': [],
}]

select_keys = list(to_tune[0].keys())

scores = ['neg_mean_absolute_error' , 'neg_mean_squared_error'] #'r2'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

selected_hypers = selected_hypers.drop(['mse', 'mae', 'mpe'], axis=1)

results = []
for col_name in y.columns:#'3h_feature_pca_0...9'
    pca_num = int(col_name.replace('3h_feature_pca_',''))
    hyper = selected_hypers.iloc[pca_num]
    features_path = os.path.join(feat_dir, f'top_features_pca_{pca_num}.txt')
    with open(features_path, 'r') as f:
        features = f.read().splitlines()

    top_features_index = pd.Index(features)

    Xfeat = X[top_features_index]
    
    id_list = []
    y_trues = []
    y_preds = []
    np.random.seed(42)
    loo = LeaveOneOut()
    y_part = y[col_name]
    for train_index, test_index in loo.split(X):
        X_train, X_test = Xfeat.iloc[train_index], Xfeat.iloc[test_index]
        y_train, y_test = y_part.iloc[train_index], y_part.iloc[test_index]
        ids = X_test.index 
        pipeline = Pipeline([
            ('scaler', StandardScaler()),  # Normalization
            ('model', XGBRegressor(n_jobs=n_jobs))  # XGBRegressor model
        ])

        model_hyperparameters = {key: value for key, value in hyper.items() if key.startswith('model__')}
        pipeline.set_params(**model_hyperparameters)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mpe = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        params = {'mse': mse, 'mae': mae, 'mpe': mpe}
        results.append(params)
        
        id_list.append(ids[0])
        y_trues.append(y_test.values[0])
        y_preds.append(y_pred[0])
    y_test_pred = pd.DataFrame({'Evlp Id No': id_list, 'y_test':y_trues, 'y_pred': y_preds})
    y_test_pred.to_csv(os.path.join(save_dir, f'{col_name}_predictions.csv'), index=False)

results_df = pd.DataFrame(results)

results_df.to_csv('results.csv', index=False)
















# %%
