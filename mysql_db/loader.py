import os
from ucimlrepo import fetch_ucirepo
import pandas as pd
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    Integer,
    Float
)
import time

MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "user": os.getenv("MYSQL_USER", "mlops_user"),
    "password": os.getenv("MYSQL_PASSWORD", "mlops_pass"),
    "database": os.getenv("MYSQL_DB", "mlops_db"),
    "port": int(os.getenv("MYSQL_PORT", "3306")),
}

metadata = MetaData()

covertype_raw = Table(
    "covertype_raw",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True)
)

def get_engine():
    url = (
        f"mysql+pymysql://{MYSQL_CONFIG['user']}:"
        f"{MYSQL_CONFIG['password']}@"
        f"{MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}/"
        f"{MYSQL_CONFIG['database']}"
    )
    return create_engine(url)


def wait_for_db(retries=10, sleep=3):

    engine = get_engine()

    for i in range(retries):
        try:
            with engine.connect():
                print("✅ DB ready")
                return
        except Exception as e:
            if i == retries - 1:
                raise RuntimeError(f"Database not reachable: {e}")

            print(f"⏳ Waiting for DB... ({i+1}/{retries})")
            time.sleep(sleep)

def load_covertype_dataset():
    """Descarga el dataset covertype desde UCI"""
    
    covertype = fetch_ucirepo(id=31)

    X = covertype.data.features
    y = covertype.data.targets

    df = pd.concat([X, y], axis=1)

    print(f"Dataset loaded: {df.shape}")

    return df


def clear_database():

    engine = get_engine()

    metadata.reflect(bind=engine)

    metadata.drop_all(bind=engine , tables=[covertype_raw])

    print("✅ Tables dropped")

def load_covertype():
    """Carga el dataset covertype a MySQL"""

    wait_for_db()

    df = load_covertype_dataset()

    engine = get_engine()

    df.to_sql(
        "covertype_raw",
        con=engine,
        if_exists="replace",
        index=False,
        method="multi",
        chunksize=5000
    )

    print(f"✅ Loaded {len(df)} rows into covertype_raw")

if __name__ == "__main__":

    wait_for_db()

    clear_database()

    load_covertype()