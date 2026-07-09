import numpy as np
import pandas as pd
from typing import Dict, List
from bisect import bisect_left


class FastSnapshotLookup:
    """
    Optimized lookup structure using Numpy arrays and Index Mapping.
    Builds in O(N log N) due to sorting, and queries in O(1) (key lookup) + O(log M) (time search).
    """

    def __init__(self, key_col: str, time_col: str, value_cols: List[str]):
        self.key_col = key_col
        self.time_col = time_col
        self.value_cols = value_cols

        # Internal storage
        self.all_keys = None  # For debug if needed
        self.all_times = None
        self.all_values = {}

        # Mapping: Key -> (Start Index, End Index)
        # Using a dict for O(1) access to key ranges
        self.key_index_map = {}

    def build(self, df: pd.DataFrame):
        """
        Builds the lookup structure efficiently using Numpy operations.
        """
        if df.empty:
            return

        # 1. Sort by Key and Time
        # Using mergesort for stability, though not strictly required
        df_sorted = df.sort_values(by=[self.key_col, self.time_col])

        # 2. Extract Numpy Arrays
        keys = df_sorted[self.key_col].astype(str).values
        self.all_times = df_sorted[self.time_col].values.astype("float64")

        for col in self.value_cols:
            if col in df_sorted.columns:
                self.all_values[col] = df_sorted[col].values.astype("float64")
            else:
                self.all_values[col] = np.full(len(df_sorted), np.nan, dtype="float64")

        # 3. Build Key Index Map
        # np.unique with return_index gives the first occurrence of each key
        unique_keys, start_indices = np.unique(keys, return_index=True)

        # We need end indices as well.
        # The end index of key[i] is the start index of key[i+1], and for the last key it's len(keys)
        # But 'unique_keys' returned by np.unique are sorted.
        # Since we sorted the dataframe by key, unique_keys logic matches the order in df_sorted?
        # Yes, df_sorted is sorted by key. So keys array looks like [A, A, B, B, C].
        # np.unique returns sorted unique keys.

        count = len(unique_keys)
        end_indices = np.empty_like(start_indices)
        end_indices[:-1] = start_indices[1:]
        end_indices[-1] = len(keys)

        # Zip into a dictionary
        # This loop runs K times (number of unique keys). K ~ 100k is fast enough in Python.
        self.key_index_map = dict(zip(unique_keys, zip(start_indices, end_indices)))

    def query_batch(
        self, keys: np.ndarray, query_times: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Batch query for multiple (key, time) pairs.
        """
        n = len(keys)
        # Pre-allocate result arrays
        results = {col: np.full(n, np.nan, dtype="float64") for col in self.value_cols}
        results[self.time_col] = np.full(n, np.nan, dtype="float64")

        # Iterate and query
        # Since we have N queries, this loop runs N times.
        # Inside is dict lookup + numpy slicing + bisect.

        # Optimizing the loop:
        # Resolving attributes to locals
        key_map = self.key_index_map
        all_times = self.all_times
        all_values = self.all_values
        time_result = results[self.time_col]
        time_col_name = self.time_col

        # value arrays
        res_arrays = {c: results[c] for c in self.value_cols}
        data_arrays = {c: all_values[c] for c in self.value_cols}
        cols = self.value_cols

        for i in range(n):
            key = str(keys[i])
            q_time = query_times[i]

            if np.isnan(q_time):
                continue

            # 1. Find Range
            indices = key_map.get(key)
            if indices is None:
                continue

            start, end = indices

            # 2. Slice Times
            # Since we only read, slicing is a view (cheap)
            times_view = all_times[start:end]

            # 3. Bisect
            # bisect_left returns insertion point to maintain order
            # We want strict past: time < q_time
            # idx = bisect_left(times, q_time)
            # times[idx] >= q_time. times[idx-1] < q_time.
            idx = bisect_left(times_view, q_time)

            if idx > 0:
                # Target found at global index: start + idx - 1
                target_global_idx = start + idx - 1

                time_result[i] = all_times[target_global_idx]
                for c in cols:
                    res_arrays[c][i] = data_arrays[c][target_global_idx]

        return results
