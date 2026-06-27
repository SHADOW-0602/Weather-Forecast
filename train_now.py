import argparse
import datetime
from noaa_client import CITIES, NOAAClient
from ml_model import HazardPredictor

# -- CLI arguments -----------------------------------------------------
parser = argparse.ArgumentParser(description="Train AeroClim multimodal models.")
parser.add_argument(
    "--city", type=str, default="SEATTLE",
    choices=sorted(CITIES),
    help="Configured city alias or NOAA station ID (default: SEATTLE)"
)
parser.add_argument(
    "--days", type=int, default=12000,
    help="Historical lookback days (default: 12000, covering the full archive)"
)
args = parser.parse_args()

# -- Data loading ------------------------------------------------------
client    = NOAAClient()
predictor = HazardPredictor()

today      = datetime.date.today()
archive_start = datetime.date(1995, 1, 1)
start_date = max(today - datetime.timedelta(days=args.days), archive_start)

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
        "Check the station file under data/noaa_stations/ and run "
        "tools/audit_data_health.py, or increase --days."
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
              f"MAE: {m.get('lstm_mae_c', 0):.2f} C  "
              f"RMSE: {m.get('lstm_rmse_c', 0):.2f} C")
    print(f"  Fusion - Accuracy: {m.get('fusion_accuracy', 0):.4f}  "
          f"AUC-ROC: {m.get('fusion_auc', 0):.4f}")
