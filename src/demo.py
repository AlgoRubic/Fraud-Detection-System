from src.predict import predict
import pandas as pd

# Load real data sample
df = pd.read_csv("data/fraudTest.csv")

sample = df.drop("is_fraud", axis=1).iloc[0].to_dict()

result = predict(sample)

print("Sample Input:\n", sample)
print("\nPrediction:\n", result)