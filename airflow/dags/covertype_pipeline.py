from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import mysql.connector
import pandas as pd
import os
import time
from sqlalchemy import create_engine
from src.utils import preprocess_data,get_data, \
    api_to_dataframe,wait_for_db,  \
    add_uuid, insert_raw, get_pending_rows, insert_processed



def insert_raw_data():

    wait_for_db()

    api_response = get_data()
    df = api_to_dataframe(api_response)
    df = add_uuid(df)
    insert_raw(df)

    #process_api_batch(api_response, "covertype_processed")

"""
def preprocess_data_for_training():

    wait_for_db()

    pending = get_pending_rows()

    if pending.empty:
        print("No new rows to process")
        return

    df_processed, _, _, _ = preprocess_data(pending)

    insert_processed(df_processed)
"""

def preprocess_data_for_training():

    wait_for_db()

    pending = get_pending_rows()

    if pending.empty:
        print("No new rows to process")
        return

    df_processed, _, _, encoders = preprocess_data(pending)

    insert_processed(df_processed)

    import joblib
    joblib.dump(encoders["onehot"], "encoders/ohe_encoder.joblib")

    print("Encoder saved")


# Definición del DAG
with DAG(
    dag_id="covertype_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["mlops", "covertype"],
) as dag:
    
    t1 = PythonOperator(
        task_id="insert_raw_data",
        python_callable=insert_raw_data,
    )
    
    t2 = PythonOperator(
        task_id="preprocess_data",
        python_callable=preprocess_data_for_training,
    )
    
    t1 >> t2