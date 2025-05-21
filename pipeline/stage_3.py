import pandas as pd
import json
import os
import requests
import numpy as np
from shapely.geometry import Point
import geopandas as gpd
from pathlib import Path
import pickle as pkl
from tqdm import tqdm
from time import sleep
from utils import get_polygon_wkt, unformat_timestamp

#---------------#
# MODIS Imagery #
#---------------#

# MCD42A2 water cover map - binarize
WATER_DICT = {
    0: 1, # shallow ocean
    1: 0, # land
    2: 0, # ocean coastlines and lake shorelines
    3: 1, # shallow inland water
    4: 1, # ephemeral water
    5: 1, # deep inland water
    6: 1, # moderate or continental ocean
    7: 1, # deep ocean
    255: 0 # fill value, treat as land for simplicity
}

def _clean_a2_data(arr):
    # Snow: 0 = no snow, 1 = snow, 255 = fill
    snow_arr = np.where((arr[0] == 255), -1, arr[0]).astype(np.float32)
    water_arr = np.vectorize(WATER_DICT.get)(arr[2]).astype(np.float32)
    return np.stack((snow_arr, water_arr), axis=0)[:,1:9,1:9]


def _clean_a4_data(arr):
    # For all MODIS bands, we're treating -1 as a fill value
    arr = np.where((arr > 30000) | (arr < 0), -10000, arr)
    arr = np.where(arr > 10000, 10000, arr)
    arr = arr / 10000.0
    return arr[:,1:9,1:9].astype(np.float32)


def _load_meta(site_path):
    with open(site_path / 'meta.json', 'r') as f:
        return json.load(f)


def create_modis_script_config(data_dir):
    data_path = Path(data_dir)
    processed_path = data_path / 'processed'
    with open(data_dir / 'config.json', 'r') as f:
        config = json.load(f)
    
    sites = os.listdir(processed_path)
    data = [_load_meta(processed_path / s) for s in sites]
    meta_df = pd.DataFrame(data)
    offset_m = config["MODIS"]["image_size_km"] * 1000 / 2
    meta_df['geometry'] = meta_df.apply(lambda row: get_polygon_wkt(row['LOCATION_LAT'], row['LOCATION_LON'], offset_m), axis=1)
    
    meta_df = meta_df[['SITE_ID', 'LOCATION_LON', 'LOCATION_LAT', 'MIN_DATE', 'MAX_DATE', 'geometry']]
    gdf = gpd.GeoDataFrame(meta_df, crs='EPSG:4326', geometry=meta_df['geometry'])
    gdf.to_file(data_path / 'sites.geojson', driver='GeoJSON')


#------------------#
# Phenocam Imagery #
#------------------#

BASE_URL = 'https://phenocam.nau.edu'
camera_url = lambda camera: f'{BASE_URL}/api/middayimages/{camera}'


def _get_all_timestamps(min_date, max_date):
    min_date_formatted = pd.to_datetime(min_date, format='%Y%m%d%H%M')
    max_date_formatted = pd.to_datetime(max_date, format='%Y%m%d%H%M')
    timestamp_range = pd.date_range(start=min_date_formatted, end=max_date_formatted, freq='30T')
    timestamp_range_int = timestamp_range.strftime('%Y%m%d%H%M').astype(int)
    return list(timestamp_range_int)


def _get_image_list(camera):
    try:
        response = requests.get(camera_url(camera))
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code}, {response.text}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None


def _download_images(image_paths, downloaded_images, phenocam_path):
    for i in image_paths:
        img_file = phenocam_path / i.split('/')[-1]
        url = BASE_URL+i
        try:
            response = requests.get(url)
            if response.status_code == 200:
                with open(img_file, "wb") as file:
                    file.write(response.content)
                downloaded_images[img_file] = True

        except requests.exceptions.RequestException as e:
            print(f"\tAn error occurred: {e}")
    
        delay = np.random.uniform(low=0.01, high=0.05)
        sleep(delay)
    

def download_phenocam_imagery(data_dir):
    data_path = Path(data_dir)
    raw_path = data_path / 'raw'
    processed_path = data_path / 'processed'
    with open(data_path / 'config.json', 'r') as f:
        config = json.load(f)
    offset_m = config['phenocam']['max_distance_from_tower_m']

    phenocam_sites = pd.read_csv(raw_path / 'meta' / 'phenocam_sites.csv')
    downloaded_images = {i: True for i in os.listdir(raw_path / 'phenocam')}
    sites = os.listdir(processed_path)

    # After every camera analysis, this function caches progress
    for site in tqdm(sites):
        with open(processed_path / site / 'meta.json', 'r') as f:
            site_meta = json.loads(f.read())
        min_date = site_meta['MIN_DATE']
        max_date = site_meta['MAX_DATE']
        lat = site_meta['LOCATION_LAT']
        lon = site_meta['LOCATION_LON']
        
        cameras = []
        allowable_poly = get_polygon_wkt(lat, lon,  offset_m)
        for i, row in phenocam_sites.iterrows():
            p = Point(row['LOCATION_LON'], row['LOCATION_LAT'])
            if allowable_poly.contains(p):
                cameras.append(row['Camera'])
        
        if os.path.exists(processed_path / site / 'phenocam.pkl'):
            with open(processed_path / site / 'phenocam.pkl', 'rb') as f:
                phenocam_lookup = pkl.load(f)
        else:
            phenocam_lookup = {d: [] for d in _get_all_timestamps(min_date, max_date)}

        for camera in cameras:
            image_list = _get_image_list(camera)
            filtered_download_list = []
            for i in image_list:
                ts = unformat_timestamp(i['imgdate'])
                if ts < min_date or max_date < ts:
                    continue
                
                img_file = i['imgpath'].split('/')[-1]
                if img_file not in phenocam_lookup[ts]:
                    phenocam_lookup[ts].append(img_file)
                
                if not downloaded_images.get(img_file, False):
                    filtered_download_list.append(i['imgpath'])
            _download_images(filtered_download_list, downloaded_images, raw_path / 'phenocam')
            with open(processed_path / site / 'phenocam.pkl', 'wb') as f:
                pkl.dump(phenocam_lookup, f)


#-----------#
# Run Stage #
#-----------#


def run_stage_3_a(data_dir, download_phenocam = False):
    data_path = Path(data_dir)
    print('Creating MODIS site metadata...')
    create_modis_script_config(data_path)
    if download_phenocam:
        print('Downloading phenocam imagery...')
        download_phenocam_imagery(data_path)

def run_stage_3_b(data_dir):
    print('Organizing MODIS data...')
    data_path = Path(data_dir)
    modis_a2_path = data_path / 'raw' / 'modis_a2'
    modis_a4_path = data_path / 'raw' / 'modis_a4'
    processed_path = data_path / 'processed'
    sites = os.listdir(processed_path)

    for site in tqdm(sites):
        a2_file = modis_a2_path / f'{site}.pkl'
        a4_file = modis_a4_path / f'{site}.pkl'
        modis_processed = {}
        if os.path.exists(processed_path / site / 'modis_a2.pkl'):
            os.remove(processed_path / site / 'modis_a2.pkl', )
        if os.path.exists(processed_path / site / 'modis_a4.pkl'):
            os.remove(processed_path / site / 'modis_a4.pkl')

        if os.path.exists(a2_file) and os.path.exists(a4_file):
            with open(a2_file, 'rb') as f:
                a2_raw = pkl.load(f)
            with open(a4_file, 'rb') as f:
                a4_raw = pkl.load(f)
            for ts in a2_raw['pixel_values'].keys():
                a4_pixels = _clean_a4_data(a4_raw['pixel_values'][ts])
                a2_pixels = _clean_a2_data(a2_raw['pixel_values'][ts])
                modis_processed[ts] = np.concatenate((a4_pixels, a2_pixels), axis=0)
        
        with open(processed_path / site / 'modis.pkl', 'wb') as f:
            pkl.dump(modis_processed, f)
