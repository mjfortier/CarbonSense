# CarbonSense Processing Pipeline

This collection of notebooks is used to transform the "raw" CarbonSense data (from source networks) into the ML-ready dataset. We present them as notebooks since they are all multi-step configurable processes and we feel it is easier to understand and modify them in this format.

There are two ways in which this pipeline can be used:
- Rerunning the pipeline and changing the config
  - This can be done to include a different variable set from the ONEFlux pipeline, or to change the gap-fill tolerance
- Adding to or modifying the raw data
  - This is possible but will require some time and effort
  - To encourage reproducibility and avoid dataset fragmentation, this should only be done if the new dataset will be released

## Rerunning the Pipeline

This is a straightforward process that can be done in under an hour on most modern desktop computers.

### 0. Acquiring the data

You should now have a symbolic link called `data` in this directory, and it should have a subdirectory structure as follows:

```
data
├── carbonsense_raw.tar.gz
└── raw
    ├── ameriflux
    ├── fluxnet
    ├── icos-2023
    ├── icos-ww
    ├── modis_a2
    └── modis_a4
```

It is now safe to remove the `tar` file with `rm data/carbonsense_raw.tar.gz`. This is not necessary to proceed, but will free up 20GB of hard drive space.

### 1. Preprocessing

Each of the four data sources (FLUXNET, Ameriflux, ICOS 2023, ICOS Warm Winter) must have their data preprocessed - we extract the relevant files, downsample any half-hourly data to hourly, filter out unneeded columns, and other small housekeeping tasks.

Every notebook prepended with `1_` is a preprocessing notebook. All four must be run. This will produce an intermediate data directory, so your data directory structure should now resemble:

```
data
├── intermediate
│   └── int_1
│       ├── ameriflux
│       ├── fluxnet
│       ├── icos-2023
│       └── icos-ww
└── raw
    └── ...
```

At this point, the EC data is ready to be fused.

### 2. Data Processing

We can now run `2_process_data.ipynb` from beginning to end. This script will fuse all EC data so that there is only a single directory per site. Note that if a site has multiple disparate temporal coverage periods, it will have separate subdirectories such as:
```
data/intermediate/int_2/CZ-BK1
├── 2004-01-01_2020-12-31
│   └── ...
└── 2022-01-01_2022-12-31
│   └── ...
```

This is expected. The final tabular files should only contain contiguous data to account for time series-based modelling. Once this script is complete, the directory structure should resemble this:

```
data
├── intermediate
│   ├── int_1
│   │   └── ...
│   └── int_2
│       ├── AR-SLu
│       ├── ...
│       └── ZM-Mon
├── meta
│   └── processed_site_meta.csv
└── raw
    └── ...
```

### 5. Incorporating MODIS Data, Normalizing Values

No, that's not a mistake. Sections 3 and 4 are not needed for this task; the `carbonsense_raw.tar.gz` file contains all the MODIS data, so notebooks `3_` and `4_` are not needed. They would only be useful if *new* EC data was being added, which is covered in the following section.

Before continuing, make any desired modifications to `normalization_config.yml`. Variable normalization ranges and maximum QC flag values can be adjusted here. Once complete, run `5_normalize_and_organize.ipynb`. This will take some time, but when it finishes your directory structure should look like this:

```
data/
├── intermediate
│   └── ...
├── meta
│   └── ...
├── processed
│   └── carbonsense
│       ├── AR-SLu
│       ├── ...
│       └── ZM-Mon
└── raw
    └── ...
```

The dataset is now ready for use.

## Adding New Data
- Needs its own directory in /data/raw
- Needs a site_data.csv
- Needs its own preprocessig script which puts it in the same format at the other preprocessing scripts

- GEE must be rerun
- This should only be done if the new dataset is being publicly released
