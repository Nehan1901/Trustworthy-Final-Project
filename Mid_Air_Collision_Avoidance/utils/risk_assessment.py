import numpy as np
from scipy.spatial.distance import cdist

def check_collision_risk(df, threshold_distance=500):
    if df.empty:
        return []

    required_cols = ["latitude", "longitude", "altitude"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError("Missing required columns in flight data.")

    positions = df[["latitude", "longitude", "altitude"]].to_numpy()
    distances = cdist(positions, positions, metric="euclidean")

    warnings = []
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if distances[i, j] < threshold_distance:
                warnings.append({
                    "Aircraft 1": df.iloc[i]["flight_id"],
                    "Aircraft 2": df.iloc[j]["flight_id"],
                    "Distance": distances[i, j]
                })

    return warnings
