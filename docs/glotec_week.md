# glotec_week.py

`glotec_week.py` is a Python script that extracts up to one week's worth of records from the source SQLite database `glotec.db` and writes those records into a new weekly chunk database file. The new database file is named using the starting timestamp provided by the user—with dashes, colons, and the "T" replaced by underscores.

## Features

- **Input Timestamp:**  
  Accepts one command-line argument: a start timestamp in ISO 8601 format (e.g., `2025-02-18T11:55:00Z`).

- **One Week Chunk:**  
  Calculates a date range from the input start timestamp to one week later (7 days).

- **Querying Data:**  
  Reads all records from the `glotec` table in `glotec.db` whose `timestamp` falls within the specified range.

- **New Database File:**  
  Writes the selected records into a new SQLite database file named `glotec_<timestamp>.db`, where `<timestamp>` is the sanitized input timestamp (dashes, colons, and the "T" replaced by underscores, with the trailing "Z" removed).  
  For example, an input of `2025-02-18T11:55:00Z` produces the output file `glotec_2025_02_18_11_55_00.db`.

- **Schema Preservation:**  
  The new database file uses the same schema as the original `glotec.db` table, with columns for:
  - `timestamp`
  - `longitude`
  - `latitude`
  - `hmF2`
  - `NmF2`
  - `quality_flag`
  
  The original `uid` values are not carried over; new auto-incremented IDs are generated in the new database.

## Usage

Run the script from the command line by providing a start timestamp in ISO 8601 format. For example:

```bash
python3 glotec_week.py 2025-02-18T11:55:00Z
