import re
from pathlib import Path

import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.preprocessing import MinMaxScaler

from utils import Logger


def __downsample(data_np: ndarray, labels_np: ndarray, sample_len: int) -> tuple[ndarray, ndarray]:
    sequence_len, num_nodes = data_np.shape

    new_len = (sequence_len // sample_len) * sample_len
    data_np = data_np[:new_len]
    labels_np = labels_np[:new_len]

    data_np = data_np.reshape(-1, sample_len, num_nodes)
    downsampled_data_np = np.median(data_np, axis=1)

    labels_np = labels_np.reshape(-1, sample_len)
    downsampled_labels_np = np.max(labels_np, axis=1).round()

    return downsampled_data_np, downsampled_labels_np


def preprocess_swat(data_path: str, processed_data_path: str, sample_len: int = 10):
    Logger.init()

    # Load data
    Logger.info(f'Loading {data_path}...')
    data_df = pd.read_excel(data_path, skiprows=[0], index_col=0)
    Logger.info(f'Loaded.')

    # Replace 'Normal' and 'Attack' with 0 and 1
    Logger.info(f'Replacing Normal and Attack with 0 and 1...')
    data_df['Normal/Attack'] = data_df['Normal/Attack'].astype(str).str.replace(r'\s+', '', regex=True).map({'Normal': 0, 'Attack': 1})
    Logger.info(f'Replaced.')

    # Fill missing values
    Logger.info(f'Fill missing values...')
    data_df.fillna(data_df.mean(), inplace=True)
    data_df.fillna(0, inplace=True)
    Logger.info(f'Filled.')

    data_df.rename(columns=lambda x: re.sub(r'\s+', '', x), inplace=True)

    # Scale data using MinMaxScaler
    Logger.info(f'Scaling data...')
    data_labels = data_df['Normal/Attack']
    data_df.drop(columns=['Normal/Attack'], inplace=True)
    data_np = MinMaxScaler(feature_range=(0, 1)).fit(data_df).transform(data_df)
    Logger.info(f'Scaled.')

    # Down-sample
    Logger.info('Down-sampling...')
    downsampled_data_np, downsampled_labels_np = __downsample(data_np, data_labels.to_numpy(), sample_len)
    data_df = pd.DataFrame(downsampled_data_np, columns=data_df.columns)
    data_df['Attack'] = downsampled_labels_np
    Logger.info('Down-sampled.')

    # Drop the first 2160 rows
    Logger.info(f'Dropping the first 2160 rows...')
    data_df = data_df.iloc[2160:]
    Logger.info(f'Dropped.')

    # Save data
    Logger.info('Saving data...')
    Path(processed_data_path).parent.mkdir(parents=True, exist_ok=True)
    data_df.to_csv(processed_data_path, index=False)
    Logger.info(f'Saved to {processed_data_path} .')
