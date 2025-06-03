import os
import numpy as np
import pandas as pd
import torch
import sqlite3
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Union, Any
from dataclasses import dataclass
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

EC_PREDICTORS = ('DOY', 'TOD', 'TA', 'P', 'RH', 'VPD', 'PA', 'CO2', 'SW_IN', 'SW_IN_POT', 'SW_OUT', 'LW_IN', 'LW_OUT',
                 'NETRAD', 'PPFD_IN', 'PPFD_OUT', 'WS', 'WD', 'USTAR', 'SWC_1', 'SWC_2', 'SWC_3', 'SWC_4', 'SWC_5',
                 'TS_1', 'TS_2', 'TS_3', 'TS_4', 'TS_5', 'WTD', 'G', 'H', 'LE',)

EC_TARGETS = ('NEE', 'GPP_DT', 'GPP_NT', 'RECO_DT', 'RECO_NT', 'FCH4')


@dataclass
class CarbonSenseLoaderConfig:
    '''Configuration for CarbonSenseV2 dataloader and preprocessor

    targets - variable selection for targets. Must be a subset of EC_TARGETS
    targets_max_qc - maximum QC flag (inclusive) to allow for target values. A lower value will result
                     in fewer usable samples, but they will be of higher quality
    predictors - variable selection for predictors. Must be a subset of EC_PREDICTORS
    predictors_max_qc - similar to targets_max_qc, but applied to predictor variables
    normalization_config - dictionary object used for normalizing variables. Custom dictionaries can
                           be supplied, but should be based on the DEFAULT_NORM template
    '''
    targets: Tuple[str] = EC_TARGETS
    predictors: Tuple[str] = EC_PREDICTORS
    use_modis: bool = True
    use_phenocam: bool = False
    context_window_length: int = 32


@dataclass
class CarbonSenseBatch:
    sites: Tuple[str] # one value for each sample
    columns: Tuple[str] # common mapping for all samples in the batch
    timestamps: Tuple
    ec_values: torch.Tensor # all eddy covariance data: (batch, context_window, values)
    modis: Tuple # all modis data: (batch, (timestamp, ndarray))
    phenocam_ir: Tuple # all phenocam infrared data: (batch, (timestamp, ndarray))
    phenocam_rgb: Tuple # all phenocam rgb data: (batch, (timestamp, ndarray))
    aux_data: Tuple # all site-level data such as elevation, igbp type: (batch, dict)

    def to(self, device: Any):
        '''
        .to(device) is provided with this dataclass as a shortcut to individually moving
        every piece of data in the class.
        '''
        self.ec_values.to(device)
        raise NotImplementedError()


class CarbonSenseDataset(Dataset):
    def __init__(self,
                 data_dir: Union[str, os.PathLike],
                 config: CarbonSenseLoaderConfig,
                 sites=None):
        self.data_path = Path(data_dir)
        self.config = config
        self.columns = ('id', 'site', 'timestamp') + tuple(self.config.predictors) + tuple(self.config.targets)

        self.window_len = self.config.context_window_length
        self.sql_file = self.data_path / 'carbonsense_v2.sql'
        self.sites = sites
        if sites == None:
            with sqlite3.connect(self.sql_file) as conn:
                result = conn.execute("SELECT DISTINCT site FROM ec_data;").fetchall()
                self.sites = [s[0] for s in result]
        self.transform = transforms.Compose([transforms.PILToTensor()])
        
        indexes = []
        print('Indexing sites...')
        target_boolean = ' OR '.join([f'{t} IS NOT NULL' for t in self.config.targets])
        with sqlite3.connect(self.sql_file) as conn:
            for site in tqdm(self.sites):
                ids = conn.execute(f'SELECT id FROM ec_data WHERE site == "{site}" AND ({target_boolean}) ORDER BY id;').fetchall()
                ids = [i[0] for i in ids]
                ids = ids[self.config.context_window_length-1:]
                indexes.extend(ids)
        self.data = np.array(indexes, dtype=np.int32)
    
    def __len__(self):
        return len(self.data)

    def _load_image(self, filename):
        with Image.open(self.data_path / 'phenocam' / filename) as img_r:
            img = img_r.convert('L' if '_IR_' in filename else 'RGB')
        return self.transform(img)
    
    def __getitem__(self, idx):
        top_index = self.data[idx]
        bottom_index = top_index - self.config.context_window_length + 1
        with sqlite3.connect(self.sql_file) as conn:
            ec_data = conn.execute(f'SELECT {",".join(self.columns)} FROM ec_data WHERE id >= {bottom_index} AND id <= {top_index} ORDER BY id;').fetchall()
            modis_result = conn.execute(f'SELECT row_id, data FROM modis_data WHERE row_id >= {bottom_index} AND row_id <= {top_index};').fetchall()
            phenocam_result = conn.execute(f'SELECT row_id, files FROM phenocam_data WHERE row_id >= {bottom_index} AND row_id <= {top_index};').fetchall()

            df = pd.DataFrame(data=ec_data, columns=self.columns)
            assert len(df['site'].unique()) == 1, f'Pulled rows from multiple sites\nTop index: {top_index}, Bottom index: {bottom_index}'
            site = df['site'].unique()[0]
            aux_result = conn.execute(f'SELECT lat, lon, elev, igbp FROM site_data WHERE site == "{site}";').fetchall()        
        
        ec_timestamps = df['timestamp'].tolist()
        ts_map = {idx: timestamp for idx, timestamp in df[['id', 'timestamp']].values}
        ec_data = torch.tensor(df.drop(columns=['id', 'site', 'timestamp']).fillna(value=np.nan).astype(np.float32).values)
        ec_cols = tuple(df.drop(columns=['id', 'site', 'timestamp']).columns)

        modis_data = []
        if self.config.use_modis:
            modis_from_bytes = lambda x: torch.tensor(np.frombuffer(x, dtype=np.float32).reshape(9,8,8))
            modis_data = [(ts_map[row_id], modis_from_bytes(bytestring)) for row_id, bytestring in modis_result]

        phenocam_ir = []
        phenocam_rgb = []
        if self.config.use_phenocam:
            for row, filetext in phenocam_result:
                files = filetext.split(',')
                phenocam_ir.extend([(ts_map[row], self._load_image(f)) for f in files if '_IR_' in f])
                phenocam_rgb.extend([(ts_map[row], self._load_image(f)) for f in files if '_IR_' not in f])
        
        return site, ec_cols, ec_timestamps, ec_data, tuple(modis_data), tuple(phenocam_ir), tuple(phenocam_rgb), aux_result[0]
    
    def collate_fn(self, batch):
        sites, ec_cols, timestamps, ec_data, modis, phenocam_ir, phenocam_rgb, aux_data = zip(*batch)
        columns = ec_cols[0] # only need to keep 1 copy of the columns
        ec_values = torch.stack(ec_data, dim=0)
        return CarbonSenseBatch(sites, columns, timestamps, ec_values, modis, phenocam_ir, phenocam_rgb, aux_data)
