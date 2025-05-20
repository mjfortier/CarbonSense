import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from pathlib import Path
import json
import math
from .utils import NpEncoder


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
    print('Cleaning site data...')
    for source in sources:
        meta_df = pd.read_csv(input_path / source / 'site_data.csv')
        site_csvs = [f for f in os.listdir(input_path / source) if f != 'site_data.csv']
        print(source)
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