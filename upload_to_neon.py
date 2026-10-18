
import os
import pandas as pd
import psycopg
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
CSV_PATH = "results/edge_processed_readings.csv"
BATCH_SIZE = 500

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL missing. Check your .env file.")

df = pd.read_csv(CSV_PATH)

df = df.rename(columns={"timestamp": "created_at"})
df["created_at"] = pd.to_datetime(
    df["created_at"], errors="coerce", utc=True
)
df = df.astype(object).where(pd.notna(df), None)

columns = [
    "reading_id", "created_at", "ecg", "bioimpedence",
    "speed", "distance", "spo2", "predicted_class",
    "class_0_probability", "class_1_probability",
    "data_source", "processing_location"
]

df = df[columns]

sql = """
INSERT INTO public.patient_readings (
    reading_id, created_at, ecg, bioimpedence, speed,
    distance, spo2, predicted_class, class_0_probability,
    class_1_probability, data_source, processing_location
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (reading_id) DO UPDATE SET
    created_at = EXCLUDED.created_at,
    ecg = EXCLUDED.ecg,
    bioimpedence = EXCLUDED.bioimpedence,
    speed = EXCLUDED.speed,
    distance = EXCLUDED.distance,
    spo2 = EXCLUDED.spo2,
    predicted_class = EXCLUDED.predicted_class,
    class_0_probability = EXCLUDED.class_0_probability,
    class_1_probability = EXCLUDED.class_1_probability,
    data_source = EXCLUDED.data_source,
    processing_location = EXCLUDED.processing_location
"""

print(f"Readings to upload: {len(df)}")

with psycopg.connect(DATABASE_URL) as conn:
    with conn.cursor() as cur:
        for start in range(0, len(df), BATCH_SIZE):
            batch = df.iloc[start:start + BATCH_SIZE]
            rows = list(batch.itertuples(index=False, name=None))
            cur.executemany(sql, rows)
            print(f"Processed {min(start + len(rows), len(df))}/{len(df)}")

print("Upload completed successfully!")
