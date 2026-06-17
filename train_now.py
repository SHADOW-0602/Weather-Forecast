import datetime
from noaa_client import NOAAClient
from ml_model import HazardPredictor

# Initialize client and predictor
client = NOAAClient()
predictor = HazardPredictor()

# Load 30 years (approx. 11,000 rows) of historical weather data for Seattle to ensure large dataset training
today = datetime.date.today()
start_date = today - datetime.timedelta(days=11000)

print("Loading Seattle weather data...")
df = client.fetch_weather_data(
    "seattle", 
    start_date.strftime("%Y-%m-%d"), 
    today.strftime("%Y-%m-%d")
)

print(f"Loaded {len(df)} rows of weather data.")

# Run training
print("Training multimodal models (LSTM + RF) with integrated Copernicus SST data...")
predictor.train_multimodal(df, progress_callback=lambda pct, msg: print(f"[{pct}%] {msg}"))
print("Models successfully trained and saved!")
