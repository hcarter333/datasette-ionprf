from datasette import hookimpl
from datasette.utils.asgi import Response
from jinja2 import Template
from jinja2 import Environment
from jinja2 import FileSystemLoader
import datetime
import math

signal_colors = {"1": "#ff004b96",
                 "0": "#ff004b96",
                 "2": "#ff0000ff",
                 "3": "#ff00a5ff",
                 "4": "#ff00ffff",
                 "5": "#ff00ff00",
                 "6": "#ffff0000",
                 "7": "#ff82004b",
                 "8": "#ffff007f",
                 "9": "#ffffffff",
                 }

def db_to_s(db):
    test_db = int(db)
    if(test_db > 32):
        return "9"
    if(test_db > 27):
        return "8"
    if(test_db > 21):
        return "7"
    if(test_db > 15):
        return "6"
    if(test_db > 8):
        return "5"
    if(test_db > 2):
        return "4"
    if(test_db >= 1):
        return "3"
    return "0"



REQUIRED_COLUMNS = {"tx_lat", "tx_lng", "rx_lat", "rx_lng", "Spotter", "dB"}


@hookimpl
def prepare_connection(conn):
    conn.create_function(
        "hello_world", 0, lambda: "Hello world!"
    )

@hookimpl
def register_output_renderer():
    print("made it into the plugin")
    return {"extension": "kml", "render": render_kml, "can_render": can_render_atom}

def render_kml(
    datasette, request, sql, columns, rows, database, table, query_name, view_name, 
    data):
    from datasette.views.base import DatasetteError
    #print(datasette.plugin_config)
    if not REQUIRED_COLUMNS.issubset(columns):
        raise DatasetteError(
            "SQL query must return columns {}".format(", ".join(REQUIRED_COLUMNS)),
            status=400,
        )
    return Response(
            get_kml(rows),
            content_type="application/vnd.google-earth.kml+xml; charset=utf-8",
            status=200,
        )


def can_render_atom(columns):
    return True
    print(str(REQUIRED_COLUMNS))
    print(str(columns))
    print(str(REQUIRED_COLUMNS.issubset(columns)))
    return REQUIRED_COLUMNS.issubset(columns)

def line_color(rst):
    if(len(str(rst)) == 3):
        return signal_colors[str(rst)[1]] 
    else:
        return signal_colors[db_to_s(rst)]

def is_qso(rst):
    if(len(str(rst)) == 3):
        return True
    else:
        return False
    
def minimum_time(rows):
    min_time = datetime.datetime.strptime('2124-02-02 00:00:00', "%Y-%m-%d %H:%M:%S")
    for row in rows:
        new_time = datetime.datetime.strptime(row['timestamp'].replace('T',' '), "%Y-%m-%d %H:%M:%S")
        if new_time < min_time:
            min_time = new_time
    print('found min_time = ' + str(min_time))
    return min_time
    

#Returns the total number of minutes before the first and last QSOs + 5
def time_span(rows):
    #find the largest time
    max_time = datetime.datetime.strptime('1968-02-02 00:00:00', "%Y-%m-%d %H:%M:%S")
    for row in rows:
        new_time = datetime.datetime.strptime(row['timestamp'].replace('T',' '), "%Y-%m-%d %H:%M:%S")
        if new_time > max_time:
            max_time = new_time
    print("max time is " + str(max_time))
    
    min_time = minimum_time(rows)
    print("min time is " + str(min_time))
    span = max_time - min_time
    print(str(span.seconds))
    mins = int(math.ceil(span.seconds/(60)))
    print('minutes ' + str(mins))
    return mins

def get_kml(rows):
    from jinja2 import Template
    map_minutes = []
    mins = time_span(rows)
    print("mins " + str(mins))
    #get the array of minutes ready to go
    map_time = minimum_time(rows)
    for minute in range(mins):
      map_time_str = str(map_time + datetime.timedelta(0,60))
      map_time_str = map_time_str.replace(' ', 'T')
      map_minutes.append(map_time_str)
      map_time = map_time + datetime.timedelta(0,60)
    for row in rows:
        print(row['timestamp'])
    with open('./plugins/templates/qso_map_header.kml') as f:
        #tmpl = Template(f.read())
        tmpl = Environment(loader=FileSystemLoader("./plugins/templates")).from_string(f.read())
        tmpl.globals['line_color'] = line_color
        tmpl.globals['is_qso'] = is_qso
    return(tmpl.render(
        kml_name = 'my first map',
        Rows = rows,
        Map_minutes = map_minutes
    ))
