import sys

sys.path.append("./")
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from vanet_env import config


def summnet_preprocess(num_content=config.NUM_CONTENT):
    path = os.path.join(os.path.dirname(__file__), "data", "SMMnet", "course-meta.csv")

    df = pd.read_csv(path, sep="\\t")

    print(df.info())

    df["catch"] = pd.to_datetime(df["catch"], format="mixed")
    df["month"] = df["catch"].dt.month
    df["day"] = df["catch"].dt.day
    df["time_only"] = df["catch"].dt.strftime("%H:%M:%S.%f")

    df["seconds"] = (
        df["catch"].dt.hour * 3600
        + df["catch"].dt.minute * 60
        + df["catch"].dt.second
        + df["catch"].dt.microsecond / 1e6
    )
    # find id last occurrence, not necessary
    # df = df.groupby("id").last().reset_index()

    scaler = MinMaxScaler()
    df["seconds_normalized"] = scaler.fit_transform(df[["seconds"]])

    # Sort by 'players'
    sorted_players_df = df.sort_values(by="players", ascending=False)

    # Sort by 'attempts'
    sorted_attempts_df = df.sort_values(by="attempts", ascending=False)

    # Sort by 'stars'
    sorted_stars_df = df.sort_values(by="stars", ascending=False)

    return df
