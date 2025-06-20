import os
import shutil
from pathlib import Path
from typing import Union
from tqdm import tqdm


class PreProcessor:
    def __init__(self,
                 data_dir: Union[str, os.PathLike],
                 source_name: str):
        self.raw_data_dir = Path(data_dir) / 'raw'
        self.preprocessed_dir = Path(data_dir) / 'preprocessed'
        self.source_name = source_name
    
    def process(self):
        """
        This method is what is called to process existing eddy covariance data into
        a format suitable for CarbonSense. The output directory should contain a
        single subdirectory per data source. Each subdirectory will contain a single
        CSV file with all the EC data, as well as a single `site_data.csv` file which
        contains all metadata for the respective EC sites (lon, lat, IGBP, etc).
        """
        raise NotImplementedError('Preprocessor does not have a process method!')


class AmerifluxPreprocessor(PreProcessor):
    def process(self):
        input_dir = self.raw_data_dir / self.source_name
        output_dir = self.preprocessed_dir / self.source_name

        def file_check(filename):
            return 'FLUXNET_SUBSET_HH' in filename and 'VARINFO' not in filename

        os.makedirs(output_dir, exist_ok=True)
        shutil.copy(input_dir / 'site_data.csv', output_dir / 'site_data.csv')
        for site in tqdm(os.listdir(input_dir / 'unzipped')):
            site_dir = input_dir / 'unzipped' / site
            files = os.listdir(site_dir)
            valid_files = [f for f in files if file_check(f)]
            if len(valid_files) == 1:
                shutil.copy(site_dir / valid_files[0], output_dir / f'{site}.csv')
            else:
                #print(f'No valid file found for {site}')
                pass


class FluxnetPreprocessorCO2_2015(PreProcessor):
    def process(self):
        input_dir = self.raw_data_dir / self.source_name
        output_dir = self.preprocessed_dir / self.source_name

        def file_check(filename):
            return 'SUBSET_HH' in filename and 'VARINFO' not in filename

        os.makedirs(output_dir, exist_ok=True)
        shutil.copy(input_dir / 'site_data.csv', output_dir / 'site_data.csv')
        for site in tqdm(os.listdir(input_dir / 'unzipped')):
            site_dir = input_dir / 'unzipped' / site
            files = os.listdir(site_dir)
            valid_files = [f for f in files if file_check(f)]
            if len(valid_files) == 1:
                shutil.copy(site_dir / valid_files[0], output_dir / f'{site}.csv')
            else:
                #print(f'No valid file found for {site}')
                pass


class FluxnetPreprocessorCH4_2015(PreProcessor):
    def process(self):
        input_dir = self.raw_data_dir / self.source_name
        output_dir = self.preprocessed_dir / self.source_name

        def file_check(filename):
            return 'CH4_HH' in filename and 'VARINFO' not in filename

        os.makedirs(output_dir, exist_ok=True)
        shutil.copy(input_dir / 'site_data.csv', output_dir / 'site_data.csv')
        for site in tqdm(os.listdir(input_dir / 'unzipped')):
            site_dir = input_dir / 'unzipped' / site
            site_name = site[4:10] # specific to this directory structure
            files = os.listdir(site_dir)
            valid_files = [f for f in files if file_check(f)]
            if len(valid_files) == 1:
                shutil.copy(site_dir / valid_files[0], output_dir / f'{site_name}.csv')
            else:
                #print(f'No valid file found for {site}')
                pass


class ICOS2023Preprocessor(PreProcessor):
    def process(self):
        input_dir = self.raw_data_dir / self.source_name
        output_dir = self.preprocessed_dir / self.source_name
        def file_check(filename):
            return 'FLUXNET_HH_L2' in filename and 'VARINFO' not in filename

        os.makedirs(output_dir, exist_ok=True)
        shutil.copy(input_dir / 'site_data.csv', output_dir / 'site_data.csv')
        for site in tqdm(os.listdir(input_dir / 'unzipped')):
            site_dir = input_dir / 'unzipped' / site
            files = os.listdir(site_dir)
            valid_files = [f for f in files if file_check(f)]
            if len(valid_files) == 1:
                shutil.copy(site_dir / valid_files[0], output_dir / f'{site}.csv')
            else:
                #print(f'No valid file found for {site}')
                pass


class ICOSWarmWinterPreprocessor(PreProcessor):
    def process(self):
        input_dir = self.raw_data_dir / self.source_name
        output_dir = self.preprocessed_dir / self.source_name

        def file_check(filename):
            return 'FULLSET_HH' in filename and 'VARINFO' not in filename

        os.makedirs(output_dir, exist_ok=True)
        shutil.copy(input_dir / 'site_data.csv', output_dir / 'site_data.csv')
        for site in tqdm(os.listdir(input_dir / 'unzipped')):
            site_dir = input_dir / 'unzipped' / site
            files = os.listdir(site_dir)
            valid_files = [f for f in files if file_check(f)]
            if len(valid_files) == 1:
                shutil.copy(site_dir / valid_files[0], output_dir / f'{site}.csv')
            else:
                #print(f'No valid file found for {site}')
                pass


REGISTER = [
    ('ameriflux', AmerifluxPreprocessor),
    ('fluxnet', FluxnetPreprocessorCO2_2015),
    ('fluxnet-ch4', FluxnetPreprocessorCH4_2015),
    ('icos-2023', ICOS2023Preprocessor),
    ('icos-ww', ICOSWarmWinterPreprocessor),
]

def run_stage_1(data_dir):
    for source_name, preprocessor in REGISTER:
        print(f'Preprocessing initiated for {source_name}...')
        preprocessor(data_dir, source_name).process()
