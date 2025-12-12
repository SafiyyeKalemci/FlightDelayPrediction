import pandas as pd

def load_flights_data():
    df = pd.read_csv("./data/flights.csv", low_memory = False)
    return df
    