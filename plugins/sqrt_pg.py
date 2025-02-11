from datasette import hookimpl
import math

@hookimpl
def prepare_connection(conn):
    conn.create_function("sqrt", 1, math.sqrt)
