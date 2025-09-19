# Crea test_db.py
import psycopg2

try:
    conn = psycopg2.connect(
        host="localhost",
        database="verifact", 
        user="postgres",
        password="verifact123"
    )
    print("Database connesso!")
    conn.close()
except Exception as e:
    print(f"Errore: {e}")