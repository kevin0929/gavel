import json
import statistics as stat
import warnings

import numpy as np
import pandas as pd
import pywt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

from utils.preprocess import filter

# filter pandas warning msg
warnings.filterwarnings("ignore")


def process(data_path: str, json_path: str, format: str) -> None:
    """_summary_

    Args:
        data_path (str): where store csv file
        json_path (str): where record time series info
    """

    # transform csv file to dataframe and read json file
    df = pd.read_csv(data_path)

    with open(json_path, "r") as file:
        time_series_dict = json.load(file)

    # dataframe rename and reserve columns we need
    df = df.rename(
        columns={
            "tickstotimestamp": "timestamp",
            "accelerometerx": "acc_x",
            "accelerometery": "acc_y",
            "accelerometerz": "acc_z",
            "gyroscopex": "gyro_x",
            "gyroscopey": "gyro_y",
            "gyroscopez": "gyro_z",
            "標記轉換": "label",
        }
    )

    # if sensor list size more than one,
    # it represent data contain multicow data.

    sensor_list = df["node_address"].unique()

    # to use band pass filter to remove noise
    df = df[
        [
            "node_address",
            "timestamp",
            "acc_x",
            "acc_y",
            "acc_z",
            "gyro_x",
            "gyro_y",
            "gyro_z",
            "label",
        ]
    ]

    df.dropna(inplace=True)

    # config sliding window data parameter
    frequency = 5
    window_size = 12 * frequency
    overlapping = 2

    data_frame = []
    label_frame = []

    # depend on sensor id and time series to make data by part
    for sensor in sensor_list:
        sensor_df = df.loc[df["node_address"] == sensor]

        sensor_df["timestamp"] = pd.to_datetime(sensor_df["timestamp"])
        sensor_df.sort_values("timestamp", ascending=True, inplace=True)

        time_series_list = time_series_dict[sensor]
        for time_series in time_series_list:
            start_time = time_series[0]
            end_time = time_series[1]

            sub_df = sensor_df.loc[
                (sensor_df["timestamp"] >= start_time)
                & (sensor_df["timestamp"] <= end_time)
            ]

            data_set = sub_df.drop(["node_address", "timestamp"], axis=1)
            y_data = data_set["label"]
            x_data = data_set.drop(["label"], axis=1)

            # normalize input data
            scaler = MinMaxScaler()
            x_data = scaler.fit_transform(x_data)

            y_data = y_data.astype(int)

            columns = ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
            x_data = pd.DataFrame(x_data, columns=columns)

            for idx in range(0, (len(x_data) - window_size), overlapping):
                a_x = x_data["acc_x"].values[idx : idx + window_size]
                a_y = x_data["acc_y"].values[idx : idx + window_size]
                a_z = x_data["acc_z"].values[idx : idx + window_size]
                g_x = x_data["gyro_x"].values[idx : idx + window_size]
                g_y = x_data["gyro_y"].values[idx : idx + window_size]
                g_z = x_data["gyro_z"].values[idx : idx + window_size]

                label = stat.mode(y_data[idx : idx + window_size])
                arr = np.asarray([a_x, a_y, a_z, g_x, g_y, g_z]).reshape(window_size, 6)
                data_frame.append(arr)
                label_frame.append(label)

    x_data = np.array(data_frame)
    y_data = np.array(label_frame)

    # if format is train, need to divide dataset into train and test
    if format == "train":
        # split dataset into train / test data
        X_train, X_test, y_train, y_test = train_test_split(
            x_data, y_data, test_size=0.2, random_state=42, shuffle=True
        )

        # use cwt to transform sensor data to image-based data
        x_train = np.ndarray(shape=(X_train.shape[0], 60, 60, 6))
        scale = range(1, 61)
        wavelet = "gaus5"
        for i in range(X_train.shape[0]):
            for j in range(0, 6):
                sig = X_train[i, :, j]
                coeff, freq = pywt.cwt(sig, scale, wavelet)
                coeff_ = coeff[:, :60]
                # print(coeff)
                x_train[i, :, :, j] = coeff_

        x_test = np.ndarray(shape=(X_test.shape[0], 60, 60, 6))
        scale = range(1, 61)
        wavelet = "gaus5"
        for i in range(X_test.shape[0]):
            for j in range(0, 6):
                sig = X_test[i, :, j]
                coeff, freq = pywt.cwt(sig, scale, wavelet)
                coeff_ = coeff[:, :60]
                x_test[i, :, :, j] = coeff_

        # save data into numpy type
        np.savez("train_data.npz", x_train=x_train, y_train=y_train)
        np.savez("test_data.npz", x_test=x_test, y_test=y_test)

    elif format == "validation":
        # split data name into file name
        filename = data_path.split(".")[0]

        x_valid = np.ndarray(shape=(x_data.shape[0], 60, 60, 6))
        scale = range(1, 61)
        wavelet = "gaus5"
        for i in range(x_data.shape[0]):
            for j in range(0, 6):
                sig = x_data[i, :, j]
                coeff, freq = pywt.cwt(sig, scale, wavelet)
                coeff_ = coeff[:, :60]
                x_valid[i, :, :, j] = coeff_

        # save data into numpy type
        np.savez(f"{filename}.npz", x_valid=x_valid, y_valid=y_data)


if __name__ == "__main__":
    data_path = "train_dataset.csv"
    json_path = "train_dataset.json"

    # start process data
    process(data_path, json_path, "train")
