from datasette import hookimpl
from datasette.utils.asgi import Response
from jinja2 import Template
from jinja2 import Environment
from jinja2 import FileSystemLoader
import datetime

REQUIRED_COLUMNS = {"Spotter", "date", "time", "park"}


@hookimpl
def prepare_connection(conn):
    conn.create_function(
        "hello_world", 0, lambda: "Hello world!"
    )

@hookimpl
def register_output_renderer():
    print("made it into the plugin")
    return {"extension": "adif", "render": render_pota_adif, "can_render": can_render_atom}

def render_pota_adif(
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
            get_pota_adif(rows),
            content_type="application/vnd.google-earth.kml+xml; charset=utf-8",
            status=200,
        )


def can_render_atom(columns):
    return True
    print(str(REQUIRED_COLUMNS))
    print(str(columns))
    print(str(REQUIRED_COLUMNS.issubset(columns)))
    return REQUIRED_COLUMNS.issubset(columns)

def minimum_time(rows):
    min_time = datetime.datetime.strptime('2124-02-02 00:00:00', "%Y-%m-%d %H:%M:%S")
    for row in rows:
        new_time = datetime.datetime.strptime(row['timestamp'].replace('T',' '), "%Y-%m-%d %H:%M:%S")
        if new_time < min_time:
            min_time = new_time
    print('found min_time = ' + str(min_time))
    return min_time
    

def maximum_time(rows):
    max_time = datetime.datetime.strptime('1968-02-02 00:00:00', "%Y-%m-%d %H:%M:%S")
    for row in rows:
        new_time = datetime.datetime.strptime(row['timestamp'].replace('T',' '), "%Y-%m-%d %H:%M:%S")
        if new_time > max_time:
            max_time = new_time
    return max_time    

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

def get_pota_adif(rows):
    from jinja2 import Template

    #Add an end time for each QSO of one minute later (for now)

    with open('./plugins/templates/pota.adif') as f:
        #tmpl = Template(f.read())
        tmpl = Environment(loader=FileSystemLoader("./plugins/templates")).from_string(f.read())
    return(tmpl.render(
        Rows = rows,
    ))
