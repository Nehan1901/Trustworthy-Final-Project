import requests
import pandas as pd
import time
import os

def get_live_flight_data():
    url = "https://opensky-network.org/api/states/all"
    
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=3)
            if response.status_code == 200:
                print("✅ Live flight data fetched.")
                return response.json()
            else:
                print(f"⚠️ API Error {response.status_code}, retrying...")
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Request failed: {e}, retrying...")
            time.sleep(1)

    print("❌ API unreachable after retries. Falling back to mock data.")
    return None

def process_flight_data(mock_path="sample_data/mock_flight_data.csv"):
    live_data = get_live_flight_data()

    if live_data and "states" in live_data:
        df_raw = pd.DataFrame(live_data["states"])

        if df_raw.empty:
            print("⚠️ No data in response. Using mock data.")
            return pd.read_csv(mock_path)

        df = df_raw.iloc[:, [0, 1, 2, 3, 5, 6, 7, 9, 10, 11]].copy()
        df.columns = [
            "icao24", "callsign", "origin_country", "time_position",
            "longitude", "latitude", "altitude", "velocity", "heading", "vertical_rate"
        ]

        df.fillna(0.0, inplace=True)

        num_cols = ["altitude", "velocity", "longitude", "latitude", "heading", "vertical_rate"]
        df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

        df_filtered = df[df["altitude"] > 10000]  # Optional altitude filter
        if df_filtered.empty:
            print("⚠️ No high-altitude flights. Returning top 10 from raw.")
            df_filtered = df.head(10)

        print(f"✅ Processed {df_filtered.shape[0]} flights.")
        return df_filtered

    else:
        print("📄 Loading mock flight data...")
        if os.path.exists(mock_path):
            return pd.read_csv(mock_path)
        else:
            raise FileNotFoundError(f"Mock data file not found: {mock_path}")
