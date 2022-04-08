import requests

test_sample = {"3_days_sum_casual": 44.0,
                "3_days_sum_registered": 499.0,
                "CasualHourBins": 1.0,
                "RegisteredHourBins": 4.0,
                "day_type": 2.0,
                "hr": 20.0,
                "hum": 0.49,
                "mnth": 11.0,
                "rolling_mean_12_hours_casual": 26.916666666666668,
                "rolling_mean_12_hours_registered": 272.6666666666667,
                "season": 4.0,
                "temp": 0.32,
                "weathersit": 1.0,
                "weekday": 1.0,
                "windspeed": 0.2537,
                "yr": 1.0,
                "holiday": 0.0}



prediction = requests.post(
    "http://127.0.0.1:52269/predict",
    headers={"content-type": "application/json"},
    json=test_sample
).text

print(f"Number of bikes predicted: {prediction}")