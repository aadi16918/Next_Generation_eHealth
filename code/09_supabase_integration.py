
import os
import pandas as pd
import psycopg
from psycopg.rows import dict_row

DATABASE_URL = os.getenv("DATABASE_URL")
CSV_PATH = "results/edge_processed_readings.csv"
BATCH_SIZE = 500

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is missing.")

df = pd.read_csv(CSV_PATH)

required = [
    "reading_id", "timestamp", "ecg", "bioimpedence",
    "speed", "distance", "spo2", "predicted_class",
    "class_0_probability", "class_1_probability",
    "data_source", "processing_location"
]

missing = [col for col in required if col not in df.columns]
if missing:
    raise ValueError(f"Missing CSV columns: {missing}")

df = df[required].copy()
df = df.rename(columns={"timestamp": "created_at"})
df["created_at"] = pd.to_datetime(
    df["created_at"], errors="coerce", utc=True
)

df = df.where(pd.notna(df), None)

sql = """
INSERT INTO patient_readings (
    reading_id, created_at, ecg, bioimpedence, speed,
    distance, spo2, predicted_class, class_0_probability,
    class_1_probability, data_source, processing_location
)
VALUES (
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
)
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
    processing_location = EXCLUDED.processing_location;
"""

columns = [
    "reading_id", "created_at", "ecg", "bioimpedence",
    "speed", "distance", "spo2", "predicted_class",
    "class_0_probability", "class_1_probability",
    "data_source", "processing_location"
]

with psycopg.connect(DATABASE_URL) as conn:
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('public.patient_readings')")
        if cur.fetchone()[0] is None:
            raise RuntimeError(
                "Table public.patient_readings not found. "
                "Create the table in Neon first."
            )

        total = len(df)
        print(f"Total readings: {total}")

        for start in range(0, total, BATCH_SIZE):
            batch = df.iloc[start:start + BATCH_SIZE]
            rows = [
                tuple(row)
                for row in batch[columns].itertuples(
                    index=False, name=None
                )
            ]
            cur.executemany(sql, rows)
            print(f"Uploaded {min(start + len(rows), total)}/{total}")

print("Upload completed successfully!")
