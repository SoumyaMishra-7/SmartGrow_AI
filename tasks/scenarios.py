SCENARIOS = {
    "balanced": {
        "base_temperature": 24.0,
        "temp_variance": 2.5,
        "base_humidity": 0.60,
        "base_light": 0.68,
        "rain_frequency": 0.25,
        "rain_chance": 0.10,
        "ideal": {"soil_moisture": 0.62, "nutrients": 0.58, "temperature": 24.0, "humidity": 0.62, "light": 0.70},
    },
    "hot_dry": {
        "base_temperature": 30.0,
        "temp_variance": 4.0,
        "base_humidity": 0.36,
        "base_light": 0.82,
        "rain_frequency": 0.08,
        "rain_chance": 0.05,
        "ideal": {"soil_moisture": 0.68, "nutrients": 0.60, "temperature": 24.0, "humidity": 0.58, "light": 0.65},
    },
    "stormy": {
        "base_temperature": 21.0,
        "temp_variance": 3.0,
        "base_humidity": 0.78,
        "base_light": 0.48,
        "rain_frequency": 0.55,
        "rain_chance": 0.18,
        "ideal": {"soil_moisture": 0.57, "nutrients": 0.56, "temperature": 23.0, "humidity": 0.68, "light": 0.60},
    },
}

DEFAULT_SCENARIO = "balanced"
