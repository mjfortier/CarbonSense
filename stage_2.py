import os
import shutil
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
import shutil
import json
import math
from utils import NpEncoder

#----------------#
# Column Merging #
#----------------#

def _default_qc_value(alias, default_flags):
    default = 0
    for k, v in default_flags.items():
        if k in alias and v > default:
            default = v
    return default


def _process_site_dataframe(df, config):
    # Get our config data in order
    timestamp_col = config['timestamp_column']
    all_columns = {}
    all_columns.update(config['meteorological_columns'])
    all_columns.update(config['soil_columns'])
    all_columns.update(config['flux_columns'])
    default_gapfill_qc_flags = config['default_gapfill_qc_flags']

    df = df.replace(-9999.0, np.nan)
    audit = {} # keep track of column sources

    # We will copy all data into a new harmonized dataframe
    h_df = df[[timestamp_col]].copy()
    h_df.rename(columns={timestamp_col: 'timestamp'}, inplace=True)
    n_rows = h_df.shape[0]
    audit['timestamp'] = [timestamp_col]
    
    for col, col_aliases in all_columns.items():
        colqc = f'{col}_QC'
        audit[col] = []

        col_np = np.array([np.nan] * n_rows, dtype=float)
        colqc_np = np.array([9999] * n_rows, dtype=float)

        for alias in col_aliases:
            if alias in df.columns:
                alias_np = df[alias].values

                aliasqc = f'{alias}_qc'
                if aliasqc not in df.columns:
                    aliasqc_np = np.array([_default_qc_value(alias, default_gapfill_qc_flags)] * n_rows, dtype=int)
                else:
                    aliasqc_np = np.nan_to_num(df[aliasqc].values, nan=_default_qc_value(alias, default_gapfill_qc_flags)).astype(int)
                
                fill_index = ((np.isnan(col_np)) & (~np.isnan(alias_np))) | ((~np.isnan(alias_np)) & (colqc_np > aliasqc_np))

                col_np[fill_index] = alias_np[fill_index]
                colqc_np[fill_index] = aliasqc_np[fill_index]
                audit[col].append(alias)

        colqc_np[colqc_np == 9999.0] = np.nan
        h_df.loc[:,col] = col_np
        h_df.loc[:,colqc] = colqc_np
        
    used_columns = [c for cs in audit.values() for c in cs]
    audit['unused'] = [c for c in df.columns if c not in used_columns]
    audit = {
        'step': 'harmonize_columns',
        'info': audit
    }
    return h_df, audit


def process_columns(input_dir, output_dir, config_file):
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    with open(config_file, 'r') as f:
        config = json.load(f)['harmonize_columns']

    sources = os.listdir(input_path)
    site_meta_dict = {}
    audit_dict = {}
    for source in sources:
        meta_df = pd.read_csv(input_path / source / 'site_data.csv')
        site_csvs = [f for f in os.listdir(input_path / source) if f != 'site_data.csv']
        print(f'  Processing {source}...')
        for csv in tqdm(site_csvs):
            site = csv[:-4]
            if site not in site_meta_dict.keys():
                os.makedirs(output_path / site, exist_ok=True)
                site_meta_dict[site] = {'COVERAGE': {}}
                audit_dict[site] = []
            
            df = pd.read_csv(input_path / source / csv)
            df, audit = _process_site_dataframe(df, config)
            time_min = df['timestamp'].min()
            time_max = df['timestamp'].max()
            df.to_csv(output_path / site / f'{source}_{time_min}_{time_max}.csv', index=False)

            site_meta = meta_df[meta_df['SITE_ID'] == site].to_dict('records')[0]
            site_meta = {k: v for k, v in site_meta.items() if k in config['meta_fields']}
            site_meta_dict[site].update(site_meta)
            site_meta_dict[site]['COVERAGE'][source] = [time_min, time_max]
            audit_dict[site].append(audit)

    # Metadata is processed after, since any site could pull from multiple sources.
    for s in site_meta_dict.keys():
        for ss in site_meta_dict[s].keys():
            if type(site_meta_dict[s][ss]) == float and math.isnan(site_meta_dict[s][ss]):
                site_meta_dict[s][ss] = None

    print('Writing metadata and audit logs...')
    for site in tqdm(site_meta_dict.keys()):
        with open(output_path / site / 'meta.json', 'w') as f:
            f.write(json.dumps(site_meta_dict[site], cls=NpEncoder))
        with open(output_path / site / 'audit.json', 'w') as f:
            f.write(json.dumps({'harmonize_columns': audit_dict[site]}))


#--------------#
# Site Merging #
#--------------#

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


#-----------#
# Run Stage #
#-----------#

def run_stage_2(data_dir, delete_intermediate = True):
    input_path = Path(data_dir) / 'preprocessed'
    intermediate_path = Path(data_dir) / 'tmp'
    output_path = Path(data_dir) / 'processed'
    config_file = Path(data_dir) / 'config.json'

    print('Parsing and reducing column space...')
    process_columns(input_path, intermediate_path, config_file)

    print('Merging site data from disparate releases...')
    merge_site_data(intermediate_path, output_path, config_file)
    if delete_intermediate:
        shutil.rmtree(intermediate_path)
    return
