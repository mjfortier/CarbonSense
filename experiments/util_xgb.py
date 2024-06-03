import pandas as pd
import os
import numpy as np
import pickle as pkl
import json
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
import random



def xgb_process_data(data_dir, train_sites, test_sites, run_dir):
    print('Processing data for XGBoost...')
    center_h = center_w = 3.5
    y, x = np.indices((8,8))
    distances = np.sqrt((y - center_h)**2 + (x - center_w)**2)
    weights = 1 / distances
    
    def average_band_values(pixels):
        c, _, _ = pixels.shape
        pixel_weights = np.tile(weights, (c,1,1))
        pixel_weights = np.where(pixels == -1.0, 0.0, pixel_weights)
        weighted_pixels = pixels * pixel_weights
        averaged_pixel_values = np.sum(weighted_pixels, axis=(1,2)) / (np.sum(pixel_weights, axis=(1,2)) + 1e-7)
        return averaged_pixel_values

    def add_imagery_to_df(df, modis, band_names):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        for band in band_names:
            df[band] = np.nan
        
        timestamps = list(df['timestamp'])
        for i, ts in zip(df.index, timestamps):
            pixels = modis.get(ts, None)
            if pixels is not None:
                avg_pixels = average_band_values(pixels[:,1:9, 1:9])
                assert avg_pixels.shape[0] == len(band_names), 'Number of bands inconsistent with provided list'
                df.loc[i, band_names] = avg_pixels
        return df


    band_names = ['MODIS_band_1', 'MODIS_band_2', 'MODIS_band_3', 'MODIS_band_4', 'MODIS_band_5', 'MODIS_band_6', 'MODIS_band_7', 'MODIS_snow', 'MODIS_water']
    targets = ['NEE_VUT_REF']


    if not os.path.exists(run_dir / 'xgb_train.csv'):
        train_datasets = []
        for site in train_sites:
            site_path = os.path.join(data_dir, site)
            for timeframe in os.listdir(site_path):
                timeframe_path = os.path.join(site_path, timeframe)
                pred_df = pd.read_csv(os.path.join(timeframe_path, 'predictors.csv'))
                targ_df = pd.read_csv(os.path.join(timeframe_path, 'targets.csv'), usecols=targets)
                df = pd.concat([pred_df, targ_df], axis=1)
                df.insert(0, 'SITE_ID', site)
                with open(os.path.join(timeframe_path, 'modis.pkl'), 'rb') as f:
                    modis_data = pkl.load(f)
                with open(os.path.join(timeframe_path, 'meta.json'), 'r') as f:
                    meta = json.load(f)
                df = add_imagery_to_df(df, modis_data, band_names)
                df = df[df[targets[0]].notna()]
                train_datasets.append((df, meta))

        train_df = pd.concat([t[0] for t in train_datasets], axis=0)
        train_df.reset_index(inplace=True, drop=True)
        train_df.to_csv(run_dir / 'xgb_train.csv', index=False)
        del(train_datasets)
        del(train_df)
        print('  train data complete')
    else:
        print('  train data already compiled')


    if not os.path.exists(run_dir / 'xgb_test.csv'):
        test_datasets = []
        for site in test_sites:
            site_path = os.path.join(data_dir, site)
            for timeframe in os.listdir(site_path):
                timeframe_path = os.path.join(site_path, timeframe)
                pred_df = pd.read_csv(os.path.join(timeframe_path, 'predictors.csv'))
                targ_df = pd.read_csv(os.path.join(timeframe_path, 'targets.csv'), usecols=targets)
                df = pd.concat([pred_df, targ_df], axis=1)
                df.insert(0, 'SITE_ID', site)
                with open(os.path.join(timeframe_path, 'modis.pkl'), 'rb') as f:
                    modis_data = pkl.load(f)
                with open(os.path.join(timeframe_path, 'meta.json'), 'r') as f:
                    meta = json.load(f)
                test_datasets.append((add_imagery_to_df(df, modis_data, band_names), meta))

        test_df = pd.concat([t[0] for t in test_datasets], axis=0)
        test_df.reset_index(inplace=True, drop=True)
        test_df.to_csv(run_dir / 'xgb_test.csv', index=False)
        del(test_datasets)
        del(test_df)
        print('  test data complete')
    else:
        print('  test data already compiled')


def xgb_hyperparam_search(run_dir, n_iter=10, target='NEE_VUT_REF'):
    param_dist = {
        'eta': [0.01, 0.05, 0.1, 0.2, 0.3],
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
        'n_estimators': [25, 50, 75, 100, 125, 150],
        'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bylevel': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'gamma': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 3, 4, 5],
        'alpha': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1],
        'lambda': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 1],
        'scale_pos_weight': [0.1, 0.5, 1, 2, 5, 10]
    }
    
    # Create the model
    xgb_model = xgb.XGBRegressor(use_label_encoder=False)

    # Define the random search
    random_search = RandomizedSearchCV(xgb_model, param_distributions=param_dist, n_iter=n_iter, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=8)

    train_df = pd.read_csv(os.path.join(run_dir, 'xgb_train.csv'), index_col=False)
    train_target = train_df[[target]]
    train_df.drop(columns=[target, 'SITE_ID', 'timestamp'], inplace=True)
    X_train = train_df.values
    y_train = train_target.values

    # Fit the random search model
    random_search.fit(X_train, y_train)

    # Best parameters found
    best_params = random_search.best_params_
    best_score = random_search.best_score_
    return best_params, best_score


def xgb_inference(run_dir):
    target_column='NEE_VUT_REF'
    train_df = pd.read_csv(os.path.join(run_dir, 'xgb_train.csv'))
    train_target = train_df[[target_column]]
    train_df.drop(columns=[target_column, 'SITE_ID', 'timestamp'], inplace=True)
    X_train = train_df.values
    y_train = train_target.values

    test_df = pd.read_csv(os.path.join(run_dir, 'xgb_test.csv'))
    test_final = test_df[['SITE_ID', 'timestamp', target_column]]
    test_df.drop(columns=[target_column, 'SITE_ID', 'timestamp'], inplace=True)
    X_test = test_df.values
    
    with open(run_dir / 'xgb.pkl', 'rb') as f:
        model = pkl.load(f)
    
    predictions = model.predict(X_test)
    test_final[f'Inferred'] = predictions
    test_final.to_csv(run_dir / 'xgb_inference.csv', index=False)


def xgb_multi_seed_inference(run_dir, params, target='NEE_VUT_REF'):
    target_column='NEE_VUT_REF'

    train_df = pd.read_csv(run_dir / 'xgb_train.csv')
    train_target = train_df[[target_column]]
    train_df.drop(columns=[target_column, 'SITE_ID', 'timestamp'], inplace=True)
    X_train = train_df.values
    y_train = train_target.values

    test_df = pd.read_csv(run_dir / 'xgb_test.csv')
    test_final = test_df[['SITE_ID', 'timestamp', target_column]]
    test_df.drop(columns=[target_column, 'SITE_ID', 'timestamp'], inplace=True)
    X_test = test_df.values

    for seed in list(range(0,100,10)):
        print(f'Training with seed: {seed}')

        model = xgb.XGBRegressor(seed=seed, **params)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        test_final[f'XGBoost_{seed}'] = predictions
        with open(os.path.join(run_dir, f'xgb_{seed}.pkl'), 'wb') as f:
            pkl.dump(model, f)
        
        del model
        
    test_final.set_index(['SITE_ID', 'timestamp'], inplace=True, drop=True)
    test_final = test_final.sort_index()
    test_final.to_csv(run_dir / 'results_xgb.csv')


