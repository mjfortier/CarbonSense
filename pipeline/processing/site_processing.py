import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
import json
from .utils import NpEncoder


def _merge_dataframes(df1, df2, all_cols):
    df_merged = pd.merge(df1,df2, on='timestamp', how='outer', suffixes=('_df1', '_df2'))
    df_final = df_merged[['timestamp']].copy()
    for col in all_cols:
        colqc = f'{col}_QC'
        df_final[[col]] = np.nan
        df_final[[colqc]] = np.nan
        df1qc = df_merged[f'{colqc}_df1']
        df2qc = df_merged[f'{colqc}_df2']

        # Merging rules:
        #   - For any row and any variable, use the one which has the lower QC value between the two sources
        #   - If it's a tie, use the value from the newer source (assumed df1 for this function)
        df1_copy_indices = ((df1qc.notna()) & (df2qc.isna())) | ((df1qc.notna()) & (df2qc.notna()) & (df1qc <= df2qc))
        df_final.loc[df1_copy_indices, col] = df_merged.loc[df1_copy_indices, f'{col}_df1']
        df_final.loc[df1_copy_indices, colqc] = df_merged.loc[df1_copy_indices, f'{colqc}_df1']
        df2_copy_indices = ((df2qc.notna()) & (df1qc.isna())) | ((df2qc.notna()) & (df1qc.notna()) & (df2qc < df1qc))
        df_final.loc[df2_copy_indices, col] = df_merged.loc[df2_copy_indices, f'{col}_df2']
        df_final.loc[df2_copy_indices, colqc] = df_merged.loc[df2_copy_indices, f'{colqc}_df2']
    return df_final.sort_values('timestamp').reset_index(drop=True)


def _merge_data(path, config):
    source_order = {s: i for i, s in enumerate(config['combine_sources']['source_priority'])}
    all_cols = list(config['harmonize_columns']['meteorological_columns'].keys()) \
             + list(config['harmonize_columns']['soil_columns'].keys()) \
             + list(config['harmonize_columns']['flux_columns'].keys())
    qc_cols = [f'{c}_QC' for c in all_cols]

    files = [f for f in os.listdir(path) if f not in ['audit.json', 'meta.json']]
    dfs = [(source_order.get(f.split('_')[0], 9999), pd.read_csv(f'{path}/{f}')) for f in files]
    dfs = [x[1] for x in sorted(dfs, key=lambda y: y[0])]
    if len(dfs) == 1:
        df = dfs[0]
    else:
        # Merge dataframes
        df = dfs.pop(0)
        while len(dfs) > 0:
            df = _merge_dataframes(df, dfs.pop(0), all_cols)

        # Fill missing timestamps. This prevents having multiple files per site for disjoint sources
        df['timestamp'] = df['timestamp']
        ts_col = pd.to_datetime(df['timestamp'], format='%Y%m%d%H%M')
        timestamp_range = pd.date_range(start=ts_col.min(), end=ts_col.max(), freq='30T')
        timestamp_range_int = timestamp_range.strftime('%Y%m%d%H%M').astype(int)
        existing_timestamps = set(df['timestamp'])
        missing_timestamps = timestamp_range_int[~timestamp_range_int.isin(existing_timestamps)]
        if len(missing_timestamps) > 0:
            missing_data = {c: [np.nan for _ in range(len(missing_timestamps))] for c in all_cols + qc_cols}
            missing_data['timestamp'] = missing_timestamps
            missing_df = pd.DataFrame(missing_data)
            df = pd.concat([df, missing_df], axis=0).sort_values('timestamp').reset_index(drop=True)
    cols = list(df.columns)
    ts_col = pd.to_datetime(df['timestamp'], format='%Y%m%d%H%M')
    df['DOY'] = ts_col.dt.dayofyear.astype(float) - 1.0
    df['TOD'] = ts_col.dt.hour.astype(float)
    df['TOD'] += 0.5 * (ts_col.dt.minute.astype(float) == 30).astype(float)
    df = df[['timestamp', 'DOY', 'TOD'] + cols[1:]]
    return df


def merge_site_data(input_dir, output_dir, config_file):
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    with open(config_file, 'r') as f:
        config = json.load(f)
    
    sites = os.listdir(input_path)
    for site in tqdm(sites):
        df = _merge_data(input_path / site, config)
        
        os.makedirs(output_path / site, exist_ok=True)
        shutil.copy(input_path / site / 'audit.json', output_path / site / 'audit.json')

        with open(input_path / site / 'meta.json', 'r') as f:
            meta = json.loads(f.read())
        meta['MIN_DATE'] = df['timestamp'][0]
        meta['MAX_DATE'] = df['timestamp'][len(df)-1]
        with open(output_path / site / 'meta.json', 'w') as f:
            f.write(json.dumps(meta,  cls=NpEncoder))
        
        df.to_csv(output_path / site / 'data.csv', index=False)
