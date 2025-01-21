import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from vanet_env import data_preprocess
import sys

sys.path.append("./")


class Caching:
    def __init__(self, caching_fps, num_content, seed):
        np.random.seed(seed)
        self.fps = caching_fps
        self.num_content = num_content

    def get_content(self, time):
        """
        Get content in content_list randomly based on time and popularity
        time is in (0, fps)
        """

        ontime_df = self.aggregated_df_list[time]
        filtered_df = ontime_df[ontime_df["id"].isin(self.content_list)]
        popularity_scores_list = filtered_df["popularity_score"].tolist()
        # Calculate the selection probabilities based on popularity scores
        # dev tag: may change to new model
        probabilities = np.array(popularity_scores_list)

        # 将前十个元素的概率调整为总和75%
        top_10_probabilities = probabilities[:10]
        top_10_probabilities = np.array(top_10_probabilities)
        top_10_probabilities /= top_10_probabilities.sum()  # 归一化
        top_10_probabilities *= 0.90  # 调整为总和90%

        # 将后90个元素的概率调整为总和25%
        remaining_probabilities = probabilities[10:]
        remaining_probabilities = np.array(remaining_probabilities)
        remaining_probabilities /= remaining_probabilities.sum()  # 归一化
        remaining_probabilities *= 0.10  # 调整为总和10%

        # 合并调整后的概率
        adjusted_probabilities = np.concatenate(
            [top_10_probabilities, remaining_probabilities]
        )

        # Select an id based on the calculated probabilities
        selected_id = np.random.choice(self.content_list, p=adjusted_probabilities)

        idx = self.content_list.index(selected_id)
        return idx

    def get_content_list(self):
        df = data_preprocess.summnet_preprocess()
        scaler = MinMaxScaler()

        # Normalize the data
        df["players_norm"] = scaler.fit_transform(df[["players"]])
        df["stars_norm"] = scaler.fit_transform(df[["stars"]])
        df["attempts_norm"] = scaler.fit_transform(df[["attempts"]])

        # Calculate the popularity score
        # df["popularity_score"] = (
        #     df["players"] * 0.5 + df["stars"] * 0.3 + df["attempts"] * 0.2
        # )

        df = df.sort_values(by="seconds_normalized", ascending=True)
        # Sort by popularity score in descending order and take top num_content
        # popularity_df = df.sort_values(by="popularity_score", ascending=False)

        # time trans into frame
        time_intervals = [(i / self.fps, (i + 1) / self.fps) for i in range(self.fps)]
        # top_ids_per_interval = {}
        aggregated_df_list = []

        for start, end in time_intervals:

            interval_records = df[
                (df["seconds_normalized"] >= start) & (df["seconds_normalized"] < end)
            ]
            # agg method 1
            # interval_records = interval_records.groupby("id").last().reset_index()

            # aggregated dup id
            aggregated_records = (
                interval_records.groupby("id")
                .agg({"stars_norm": "max", "id": "count"})
                .rename(columns={"id": "count"})
                .reset_index()
            )
            aggregated_records["time_interval"] = f"{start}-{end}"

            aggregated_records = aggregated_records.sort_values(
                by="count", ascending=False
            ).reset_index(drop=True)

            aggregated_records["count_norm"] = scaler.fit_transform(
                aggregated_records[["count"]]
            )
            aggregated_records["popularity_score"] = (
                aggregated_records["count_norm"] * 0.9
                + aggregated_records["stars_norm"] * 0.1
            )

            aggregated_df_list.append(aggregated_records)

            # top_10_ids = (
            #     interval_records.nlargest(10, "popularity_score")["id"]
            #     .unique()
            #     .tolist()
            # )
            # top_ids_per_interval[f"{start}-{end}"] = top_10_ids

        self.aggregated_df_list = aggregated_df_list
        self.aggregated_df = pd.concat(aggregated_df_list).reset_index(drop=True)

        # top n selected for each time period (sorted based on popularity_score), intersected with 100 different ids and kept in original order
        n = self.num_content * self.fps
        top_ids_per_interval = []

        for start, end in time_intervals:
            interval_records = self.aggregated_df[
                self.aggregated_df["time_interval"] == f"{start}-{end}"
            ]
            top_n_ids = interval_records.nlargest(n, "popularity_score")["id"].tolist()
            top_ids_per_interval.extend(top_n_ids)

        # Get 100 different ids and keep the original order
        unique_top_ids_ordered = []
        seen_ids = set()

        for id in top_ids_per_interval:
            if id not in seen_ids:
                unique_top_ids_ordered.append(id)
                seen_ids.add(id)
            if len(unique_top_ids_ordered) == self.num_content:
                break

        self.content_list = unique_top_ids_ordered

        return self.content_list, self.aggregated_df_list, self.aggregated_df
