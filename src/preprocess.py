import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    df = df.copy()

    # Handle missing values
    df.fillna(df.median(numeric_only=True), inplace=True)

    # Example feature engineering
    df['amount_log'] = df['amount'].apply(lambda x: 0 if x <= 0 else x)

    # Scaling
    scaler = StandardScaler()
    df[['amount']] = scaler.fit_transform(df[['amount']])

    return df, scaler