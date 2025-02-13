import pandas as pd
import numpy as np
import os

def merge_parquet_files(input_dir: str, output_file: str):
    parquet_files = [
        os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.parquet')
    ]

    if not parquet_files:
        print("Non file")
        return


    dataframes = []
    for file in parquet_files:
        try:
            df = pd.read_parquet(file)
            dataframes.append(df)
            print(f"read success: {file}")
        except Exception as e:
            print(f"read {file} error: {e}")

    merged_df = pd.concat(dataframes, ignore_index=True)

    try:
        merged_df.to_parquet(output_file, index=False)
        print(f"merge success: {output_file}")
    except Exception as e:
        print(f"save {output_file} error: {e}")



def get_Xy_memmap(df, window_size, memmap_filename_X, memmap_filename_y, forecast_steps = [6, 12, 24, 48, 72, 168]):
    df_as_np = df.to_numpy()
    data_len = len(df_as_np) - window_size - max(forecast_steps)  
    num_features = df.shape[1]

    X_memmap = np.memmap(memmap_filename_X, dtype='float32', mode='w+', shape=(data_len, window_size, num_features))
    y_memmap = np.memmap(memmap_filename_y, dtype='float32', mode='w+', shape=(data_len, len(forecast_steps)))

    for i in range(data_len):
        X_memmap[i] = df_as_np[i:i+window_size]  
        y_memmap[i] = [df_as_np[i+window_size+step][0] for step in forecast_steps]  

    return X_memmap, y_memmap

def normalization(X, mean, std):
    X[:, :, 0] = (X[:, :, 0] - mean) / std  
    return X


def load_real_data(file_path: str, window_size: int):
    df = pd.read_parquet(file_path)
    df['hourofday'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    
    df['Day sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['Day cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['Hour sin'] = np.sin(2 * np.pi * df['hourofday'] / 24)
    df['Hour cos'] = np.cos(2 * np.pi * df['hourofday'] / 24)   

    df = df.drop(columns=['dayofweek', 'hourofday'], axis=1)

    X, y = get_Xy_memmap(df, window_size, "X_data.dat", "y_data.dat")

    X_train, y_train = X[:25000], y[:25000]
    X_val, y_val = X[25000:30000], y[25000:30000]
    X_test, y_test = X[30000:], y[30000:]

    training_mean = np.mean(X_train[:, :, 0])
    training_std = np.std(X_train[:, :, 0])

    X_train = normalization(X_train, training_mean, training_std)
    X_val = normalization(X_val, training_mean, training_std)
    X_test = normalization(X_test, training_mean, training_std)

    return X_train, X_val, X_test, y_train, y_val, y_test


def load_synthetiques_data(file_path: str, window_size: int):
    df = pd.read_parquet(file_path).iloc[:70128]
    df['Day sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['Day cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['Hour sin'] = np.sin(2 * np.pi * df['hourofday'] / 24)
    df['Hour cos'] = np.cos(2 * np.pi * df['hourofday'] / 24)   

    df = df.drop(columns=['dayofweek', 'hourofday'], axis=1)

    X, y = get_Xy_memmap(df, window_size, "X_data.dat", "y_data.dat")

    X_train, y_train = X[:40000], y[:40000]
    X_val, y_val = X[40000:55000], y[40000:55000]
    X_test, y_test = X[55000:], y[55000:]

    training_mean = np.mean(X_train[:, :, 0])
    training_std = np.std(X_train[:, :, 0])

    X_train = normalization(X_train, training_mean, training_std)
    X_val = normalization(X_val, training_mean, training_std)
    X_test = normalization(X_test, training_mean, training_std)

    return X_train, X_val, X_test, y_train, y_val, y_test

def load_base_augmented_data(file_path: str, window_size: int):
    df = pd.read_parquet(file_path)
    df['Day sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['Day cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['Hour sin'] = np.sin(2 * np.pi * df['hourofday'] / 24)
    df['Hour cos'] = np.cos(2 * np.pi * df['hourofday'] / 24)   

    df = df.drop(columns=['dayofweek', 'hourofday'], axis=1)

    X, y = get_Xy_memmap(df, window_size, "X_data.dat", "y_data.dat")

    X_train, y_train = X[:40000], y[:40000]
    X_val, y_val = X[40000:55000], y[40000:55000]
    X_test, y_test = X[55000:], y[55000:]

    training_mean = np.mean(X_train[:, :, 0])
    training_std = np.std(X_train[:, :, 0])

    X_train = normalization(X_train, training_mean, training_std)
    X_val = normalization(X_val, training_mean, training_std)
    X_test = normalization(X_test, training_mean, training_std)

    return X_train, X_val, X_test, y_train, y_val, y_test





