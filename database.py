import sqlite3
import pandas as pd
import os
from contextlib import contextmanager

DATABASE_PATH = "experiments.db"

@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect(DATABASE_PATH)
    try:
        yield conn
    finally:
        conn.close()

def initialize_database():
    """Initialize the database with required tables and columns"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Create experiments table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tuning_method TEXT,
                best_score REAL,
                time_sec REAL,
                memory_bytes INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Check if dataset column exists, if not add it
        cursor.execute("PRAGMA table_info(experiments)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'dataset' not in columns:
            cursor.execute("ALTER TABLE experiments ADD COLUMN dataset TEXT")
        
        if 'model' not in columns:
            cursor.execute("ALTER TABLE experiments ADD COLUMN model TEXT")
        
        conn.commit()

def get_all_experiments():
    """Retrieve all experiments from the database"""
    with get_db_connection() as conn:
        query = """
            SELECT id, tuning_method, best_score, time_sec, memory_bytes, 
                   dataset, model, created_at
            FROM experiments
            ORDER BY created_at DESC
        """
        df = pd.read_sql_query(query, conn)
        return df

def add_experiment(tuning_method, best_score, time_sec, memory_bytes=None, dataset=None, model=None):
    """Add a new experiment to the database"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO experiments (tuning_method, best_score, time_sec, memory_bytes, dataset, model)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (tuning_method, best_score, time_sec, memory_bytes, dataset, model))
        conn.commit()
        return cursor.lastrowid

def delete_experiment(experiment_id):
    """Delete an experiment from the database"""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,))
        conn.commit()
        return cursor.rowcount > 0

def update_experiment(experiment_id, **kwargs):
    """Update an experiment in the database"""
    if not kwargs:
        return False
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        # Build dynamic update query
        set_clause = ", ".join([f"{key} = ?" for key in kwargs.keys()])
        values = list(kwargs.values()) + [experiment_id]
        
        cursor.execute(f"""
            UPDATE experiments 
            SET {set_clause}
            WHERE id = ?
        """, values)
        
        conn.commit()
        return cursor.rowcount > 0
