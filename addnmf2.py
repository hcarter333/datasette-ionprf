#!/usr/bin/env python3
import sqlite3

def add_nmF2_column(db_filename="glotec.db"):
    """
    Connects to the SQLite database file and adds a column 'NmF2' of type REAL
    to the table 'glotec'. This column is intended to hold numbers formatted like:
    341443542533.4619.
    """
    try:
        conn = sqlite3.connect(db_filename)
        cur = conn.cursor()
        # ALTER TABLE to add a new column. Note: ALTER TABLE in SQLite can only add columns.
        cur.execute("ALTER TABLE glotec ADD COLUMN NmF2 REAL;")
        conn.commit()
        print("Column 'NmF2' added successfully to the table 'glotec'.")
    except sqlite3.OperationalError as e:
        # Likely the column already exists or there is another operational error.
        print(f"OperationalError: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    add_nmF2_column()
