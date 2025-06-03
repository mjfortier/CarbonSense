import os
import sqlite3
import pickle as pkl
import numpy as np
import pandas as pd
from pathlib import Path
from functools import partial
from dataclasses import dataclass, field
from typing import Dict, Tuple, Union
from dataclasses import dataclass
from multiprocessing import Pool
from tqdm import tqdm
import json
from PIL import Image

EC_PREDICTORS = ('DOY', 'TOD', 'TA', 'P', 'RH', 'VPD', 'PA', 'CO2', 'SW_IN', 'SW_IN_POT', 'SW_OUT', 'LW_IN', 'LW_OUT',
                 'NETRAD', 'PPFD_IN', 'PPFD_OUT', 'WS', 'WD', 'USTAR', 'SWC_1', 'SWC_2', 'SWC_3', 'SWC_4', 'SWC_5',
                 'TS_1', 'TS_2', 'TS_3', 'TS_4', 'TS_5', 'WTD', 'G', 'H', 'LE',)

EC_TARGETS = ('NEE', 'GPP_DT', 'GPP_NT', 'RECO_DT', 'RECO_NT', 'FCH4')

DEFAULT_NORM = {
    # Predictors
    'DOY': {'cyclic': True, 'norm_max': 366.0, 'norm_min': 0.0},
    'TOD': {'cyclic': True, 'norm_max': 24.0, 'norm_min': 0.0},
    'TA': {'cyclic': False, 'norm_max': 80.0, 'norm_min': -80.0},
    'P': {'cyclic': False, 'norm_max': 50.0, 'norm_min': 0.0},
    'RH': {'cyclic': False, 'norm_max': 100.0, 'norm_min': 0.0},
    'VPD': {'cyclic': False, 'norm_max': 110.0, 'norm_min': 0.0},
    'PA': {'cyclic': False, 'norm_max': 110.0, 'norm_min': 0.0},
    'CO2': {'cyclic': False, 'norm_max': 750.0, 'norm_min': 0.0},
    'SW_IN': {'cyclic': False, 'norm_max': 1500.0, 'norm_min': -1500.0},
    'SW_IN_POT': {'cyclic': False, 'norm_max': 1500.0, 'norm_min': -1500.0},
    'SW_OUT': {'cyclic': False, 'norm_max': 500.0, 'norm_min': -500.0},
    'LW_IN': {'cyclic': False, 'norm_max': 1000.0, 'norm_min': -1000.0},
    'LW_OUT': {'cyclic': False, 'norm_max': 1000.0, 'norm_min': -1000.0},
    'NETRAD': {'cyclic': False, 'norm_max': 1000.0, 'norm_min': -1000.0},
    'PPFD_IN': {'cyclic': False, 'norm_max': 2500.0, 'norm_min': -2500.0},
    'PPFD_OUT': {'cyclic': False, 'norm_max': 1000.0, 'norm_min': -1000.0},
    'WS': {'cyclic': False, 'norm_max': 100.0, 'norm_min': -100.0},
    'WD': {'cyclic': True, 'norm_max': 360.0, 'norm_min': 0.0},
    'USTAR': {'cyclic': False, 'norm_max': 4.0, 'norm_min': -4.0},
    'SWC_1': {'cyclic': False, 'norm_max': 0.0, 'norm_min': 100.0},
    'SWC_2': {'cyclic': False, 'norm_max': 0.0, 'norm_min': 100.0},
    'SWC_3': {'cyclic': False, 'norm_max': 0.0, 'norm_min': 100.0},
    'SWC_4': {'cyclic': False, 'norm_max': 0.0, 'norm_min': 100.0},
    'SWC_5': {'cyclic': False, 'norm_max': 0.0, 'norm_min': 100.0},
    'TS_1': {'cyclic': False, 'norm_max': 40.0, 'norm_min': -40.0},
    'TS_2': {'cyclic': False, 'norm_max': 40.0, 'norm_min': -40.0},
    'TS_3': {'cyclic': False, 'norm_max': 40.0, 'norm_min': -40.0},
    'TS_4': {'cyclic': False, 'norm_max': 40.0, 'norm_min': -40.0},
    'TS_5': {'cyclic': False, 'norm_max': 40.0, 'norm_min': -40.0},
    'WTD': {'cyclic': False, 'norm_max': -3.0, 'norm_min': 3.0},
    'G': {'cyclic': False, 'norm_max': 700.0, 'norm_min': -700.0},
    'H': {'cyclic': False, 'norm_max': 700.0, 'norm_min': -700.0},
    'LE': {'cyclic': False, 'norm_max': 700.0, 'norm_min': -700.0},

    # Carbon fluxes
    'NEE': {'cyclic': False, 'norm_max': 50.0, 'norm_min': -50.0},
    'GPP_DT': {'cyclic': False, 'norm_max': 40.0, 'norm_min': -40.0},
    'GPP_NT': {'cyclic': False, 'norm_max': 40.0, 'norm_min': -40.0},
    'RECO_DT': {'cyclic': False, 'norm_max': 30.0, 'norm_min': -30.0},
    'RECO_NT': {'cyclic': False, 'norm_max': 30.0, 'norm_min': -30.0},
    'FCH4': {'cyclic': False, 'norm_max': 800.0, 'norm_min': -800.0}
}


@dataclass
class CarbonSenseStage4Config:
    '''Configuration for CarbonSenseV2 final output

    targets - variable selection for targets. Must be a subset of EC_TARGETS
    targets_max_qc - maximum QC flag (inclusive) to allow for target values. A lower value will result
                     in fewer usable samples, but they will be of higher quality
    predictors - variable selection for predictors. Must be a subset of EC_PREDICTORS
    predictors_max_qc - similar to targets_max_qc, but applied to predictor variables
    normalization_config - dictionary object used for normalizing variables. Custom dictionaries can
                           be supplied, but should be based on the DEFAULT_NORM template
    phenocam_size - the edge size used to resize phenocam imagery
    '''
    targets: Tuple[str] = EC_TARGETS
    targets_max_qc: int = 2
    predictors: Tuple[str] = EC_PREDICTORS
    predictors_max_qc: int = 2
    normalize_predictors: bool = True
    normalize_targets: bool = False
    normalization_config: Dict = field(default_factory = lambda: (DEFAULT_NORM))
    phenocam_size: int = 256


def _resize_img(img_dir, target_dir, target_size, image_file):
    with Image.open(img_dir / image_file) as img:
        img_resized = img.resize((target_size,target_size), Image.LANCZOS)
        img_resized.save(target_dir / image_file)


def resize_phenocam(
        raw_data_dir: Union[str, os.PathLike],
        output_dir: Union[str, os.PathLike],
        config: CarbonSenseStage4Config,
        num_workers: int = 8):
    """Resizes all phenocam imagery and places it in a new directory

    Args:
        raw_data_dir: Directory of original images.
        output_dir: Directory for resized images.
        config: Config used for processing.
        num_workers: Number of processes to spawn.
    """
    img_dir = Path(raw_data_dir) / 'raw' / 'phenocam'
    target_dir = Path(output_dir) / 'phenocam'
    os.makedirs(target_dir, exist_ok=True)

    resize_img_partial = partial(_resize_img, img_dir, target_dir, config.phenocam_size)
    print('Resizing phenocam imagery...')
    images = os.listdir(img_dir)
    with Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap_unordered(resize_img_partial, images), total=len(images)))


def _create_carbonsense_tables(conn: sqlite3.Connection, config: CarbonSenseStage4Config):
    columns =  config.predictors + config.targets
    column_spec = ',\n        '.join([f'{c} REAL' for c in columns])
    create_tables_statement = f"""
    DROP TABLE IF EXISTS site_data;
    DROP TABLE IF EXISTS phenocam_data;
    DROP TABLE IF EXISTS modis_data;
    DROP TABLE IF EXISTS ec_data;
    CREATE TABLE ec_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        site TEXT NOT NULL,
        timestamp INTEGER NOT NULL,
        {column_spec}
    );
    CREATE INDEX idx_ec_id ON ec_data(id);
    CREATE INDEX idx_ec_site ON ec_data(site);

    CREATE TABLE modis_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        row_id INTEGER,
        data BLOB,
        FOREIGN KEY(row_id) REFERENCES ec_data(id)
    );
    CREATE INDEX idx_modis_row_id ON modis_data(row_id);

    CREATE TABLE phenocam_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        row_id INTEGER,
        files TEXT,
        FOREIGN KEY(row_id) REFERENCES ec_data(id)
    );
    CREATE INDEX idx_phenocam_row_id ON phenocam_data(row_id);

    CREATE TABLE site_data (
        site TEXT PRIMARY KEY,
        lat REAL,
        lon REAL,
        elev REAL,
        igbp TEXT
    );
    CREATE INDEX idx_site_data_site ON site_data(site);
    """
    conn.executescript(create_tables_statement)


def _process_dataframe(df_raw: pd.DataFrame, config: CarbonSenseStage4Config):
    df = df_raw.copy()
    for pred in config.predictors:
        if pred == 'DOY' or pred == 'TOD':
            continue
        df.loc[df[f'{pred}_QC'] > config.predictors_max_qc, pred] = np.nan
    for targ in config.targets:
        df.loc[df[f'{targ}_QC'] > config.targets_max_qc, targ] = np.nan

    # Filter variables (and get rid of QC columns)
    df = df[['timestamp'] + list(config.predictors) + list(config.targets)]

    # Min-max normalization
    if config.normalize_predictors:
        for pred in config.predictors:
            vmax = config.normalization_config[pred]['norm_max']
            vmin = config.normalization_config[pred]['norm_min']
            vmid = (vmax + vmin) / 2
            vrange = vmax - vmin
            cyclic = config.normalization_config[pred]['cyclic']
            if cyclic:
                vrange /= 2

            df.loc[~df[pred].between(vmin, vmax), pred] = np.nan
            df[pred] = (df[pred] - vmid) / vrange

    if config.normalize_targets:
        for targ in config.targets:
            vmax = config.normalization_config[targ]['norm_max']
            vmin = config.normalization_config[targ]['norm_min']
            vmid = (vmax + vmin) / 2
            vrange = vmax - vmin
            cyclic = config.normalization_config[targ]['cyclic']
            if cyclic:
                vrange /= 2

            df.loc[~df[targ].between(vmin, vmax), targ] = np.nan
            df[targ] = (df[targ] - vmid) / vrange
    return df


def process_data_sql(
        data_dir: Union[str, os.PathLike],
        output_dir: Union[str, os.PathLike],
        config: CarbonSenseStage4Config):
    
    processed_data_path = Path(data_dir) / 'processed'
    output_path = Path(output_dir)
    
    sites = os.listdir(processed_data_path)
    print('Preparing site data for SQL...')
    with sqlite3.connect(output_path / 'carbonsense_v2.sql') as conn:
        _create_carbonsense_tables(conn, config)
        for site in tqdm(sites):
            df = pd.read_csv(processed_data_path / site / 'data.csv')
            df_processed = _process_dataframe(df, config)
            df_processed['site'] = site
            df_processed.to_sql('ec_data', conn, if_exists='append', index=False)

            timestamp_row_pairs = conn.execute(f'SELECT id, timestamp FROM ec_data WHERE site = "{site}";').fetchall()
            timestamp_map = {ts: ind for ind, ts in timestamp_row_pairs}

            modis_file = processed_data_path / site / 'modis.pkl'
            if os.path.exists(modis_file):
                with open(modis_file, 'rb') as f:
                    modis_dict = pkl.load(f)
                modis_data = {'row_id': [], 'data': []}
                for ts, nparr in modis_dict.items():
                    if ts not in timestamp_map.keys() or len(nparr) == 0:
                        continue
                    modis_data['row_id'].append(timestamp_map[ts])
                    modis_data['data'].append(nparr.tobytes())
                df_modis = pd.DataFrame(modis_data)
                df_modis.to_sql('modis_data', conn, if_exists='append', index=False)
                del modis_file, modis_dict, modis_data, df_modis

            phenocam_file = processed_data_path / site / 'phenocam.pkl'
            if os.path.exists(phenocam_file):
                with open(phenocam_file, 'rb') as f:
                    phenocam_dict = pkl.load(f)
                phenocam_data = {'row_id': [], 'files': []}
                for ts, filelist in phenocam_dict.items():
                    checked_filelist = _check_phenocam_files(filelist, output_path)
                    if ts not in timestamp_map.keys() or len(checked_filelist) == 0:
                        continue
                    phenocam_data['row_id'].append(timestamp_map[ts])
                    phenocam_data['files'].append(','.join(checked_filelist))
                df_phenocam = pd.DataFrame(phenocam_data)
                df_phenocam.to_sql('phenocam_data', conn, if_exists='append', index=False)
                del phenocam_file, phenocam_dict, phenocam_data, df_phenocam
            del df, df_processed, timestamp_row_pairs, timestamp_map

            meta_file = processed_data_path / site / 'meta.json'
            with open(meta_file, 'r') as f:
                meta = json.load(f)
                site_row = {
                    'site': meta['SITE_ID'],
                    'lat': meta['LOCATION_LAT'],
                    'lon': meta['LOCATION_LON'],
                    'elev': meta['LOCATION_ELEV'],
                    'igbp': meta['IGBP']
                }
            conn.execute('INSERT INTO site_data VALUES (:site, :lat, :lon, :elev, :igbp)', site_row)
        return


def _check_phenocam_files(filelist: list, output_path: os.PathLike):
    checked_filelist = []
    for file in filelist:
        if os.path.exists(output_path/'phenocam'/file):
            checked_filelist.append(file)
    return checked_filelist


def run_stage_4(
        data_dir: Union[str, os.PathLike],
        output_name: str = 'carbonsense_v2',
        config: CarbonSenseStage4Config = CarbonSenseStage4Config(),
        num_workers: int = 8):
    """Converts the raw CarbonSenseV2 dataset into a fully processed version
    ready for use in a dataloader.

    Args:
        data_dir: The location of CarbonSenseV2 preprocessed dataset.
        output_dir: Location of final CarbonSenseV2 dataset.
        config (optional): CarbonSenseStage4Config as described above.
        num_workers (optional): Number of processes to use in post processing.
    """
    data_path = Path(data_dir)
    output_path = data_path / output_name
    
    os.makedirs(output_path, exist_ok=True)
    #resize_phenocam(data_path, output_path, config, num_workers)
    process_data_sql(data_path, output_path, config)

    return
