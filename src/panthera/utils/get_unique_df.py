import pandas as pd
import hashlib
from typing import Any, List
from pandas.util import hash_pandas_object  # type: ignore


def get_unique_df(dfs: List[pd.DataFrame]) -> List[pd.DataFrame]:
    """Returns a unique list of DataFrames.
    Guarantees sensitivity to Data, Index, Column Names, and Data Types.
    """
    unique_dfs = []
    seen_hashes = set()

    for df in dfs:
        # 1. Hash core data with column names stripped out
        df_renamed = df.set_axis(range(len(df.columns)), axis="columns")

        # Use Any to bypass pyright's confusion about hash_pandas_object and avoid
        # 'Appender' or 'Substitution' class type resolution issues.
        hashes: Any = hash_pandas_object(df_renamed, index=True)  # type: ignore
        data_hash: bytes = hashes.to_numpy().tobytes()

        # 2. Column names and dtypes as explicit, stable byte strings
        cols_bytes = str(list(df.columns)).encode("utf-8")
        dtypes_bytes = str(list(df.dtypes)).encode("utf-8")

        # 3. Cryptographically combine all three dimensions
        hasher = hashlib.sha256()
        hasher.update(data_hash)
        hasher.update(cols_bytes)
        hasher.update(dtypes_bytes)

        fingerprint = hasher.hexdigest()
        if fingerprint not in seen_hashes:
            seen_hashes.add(fingerprint)
            unique_dfs.append(df)

    return unique_dfs
