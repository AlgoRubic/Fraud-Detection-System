from src.predict import predict

sample = {
    "amt": 100.5,
    "category": "shopping",
    "gender": "M",
    "city": "Delhi",
    "state": "DL",
    "lat": 28.6,
    "long": 77.2,
    "city_pop": 500000,
    "unix_time": 1234567890,
    "merchant": "abc_store",
    "job": "engineer"
}

result = predict(sample)
print(result)