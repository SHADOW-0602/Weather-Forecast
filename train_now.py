import argparse
import datetime
from noaa_client import NOAAClient
from ml_model import HazardPredictor

# -- CLI arguments -----------------------------------------------------
parser = argparse.ArgumentParser(description="Train AeroClim multimodal models.")
parser.add_argument(
    "--city", type=str, default="SEATTLE",
    choices=["SEATTLE", "NEW_YORK", "PHOENIX"],
    help="Station ID to train on (default: SEATTLE)"
)
parser.add_argument(
    "--days", type=int, default=10000,
    help="Number of historical days to load (default: 10000 ~ 27 years)"
)
args = parser.parse_args()

# -- Data loading ------------------------------------------------------
client    = NOAAClient()
predictor = HazardPredictor()

today      = datetime.date.today()
start_date = today - datetime.timedelta(days=args.days)

print(f"[train_now] Loading {args.days} days of weather data for {args.city} ...")
df = client.fetch_weather_data(
    args.city,
    start_date.strftime("%Y-%m-%d"),
    today.strftime("%Y-%m-%d"),
)
print(f"[train_now] Loaded {len(df):,} rows  |  "
      f"cols: {list(df.columns)}")

if len(df) < 200:
    raise RuntimeError(
        f"Only {len(df)} rows returned - not enough to train. "
        "Check that data/seattle.csv / data/new_york.csv / data/phoenix.csv exists "
        "or increase --days."
    )

# -- Training ----------------------------------------------------------
def _progress(pct: int, msg: str) -> None:
    bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
    print(f"  [{bar}] {pct:3d}%  {msg}")

print("\n[train_now] Starting multimodal training ...")
print("  Models: LSTM (temperature) + Random Forest (risk) + Late Fusion\n")

predictor.train_multimodal(df, progress_callback=_progress)

print("\n[train_now] [OK] Training complete!")
print("  Saved to: saved_models/")
print("    lstm_final.keras")
print("    fusion_final.keras")
print("    rf_model.pkl")
print("    scaler_lstm.pkl  |  scaler_rf.pkl")
print("    metrics.json")

# -- Print summary metrics ---------------------------------------------
if predictor.metrics:
    m = predictor.metrics
    print("\n[train_now] Performance summary (test set):")
    print(f"  RF  - Accuracy: {m.get('rf_accuracy', 0):.4f}  "
          f"AUC-ROC: {m.get('rf_auc', 0):.4f}")
    if m.get("lstm_r2", 0) != 0:
        print(f"  LSTM- R2:       {m.get('lstm_r2', 0):.4f}  "
              f"MSE: {m.get('lstm_mse', 0):.4f}")
    print(f"  Fusion - Accuracy: {m.get('fusion_accuracy', 0):.4f}  "
          f"AUC-ROC: {m.get('fusion_auc', 0):.4f}")
