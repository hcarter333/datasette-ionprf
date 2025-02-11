from datasette import hookimpl
from datasette.utils.asgi import Response
from jinja2 import Template
from jinja2 import Environment
from jinja2 import FileSystemLoader
import datetime
import math
#from ionodata import get_f2m
#from earthmid import midpoint_lng
#from earthmid import midpoint_lat


import math
deg2rad = math.pi/180
rad2deg = 180/math.pi

def cartesian_x(f,l):
    #f = latitude, l = longitude
    return (math.cos(f*deg2rad)*math.cos(l*deg2rad))

def cartesian_y(f,l):
    #f = latitude, l = longitude
    return (math.cos(f*deg2rad)*math.sin(l*deg2rad))

def cartesian_z(f,l):
    #f = latitude, l = longitude
    return (math.sin(f*deg2rad))

def spherical_lat(x,y,z):
    r = math.sqrt(x*x + y*y)
    #Omitting the special cases because points will always 
    #be separated for this application
    return (math.atan2(z, r)*rad2deg) # return degrees
    
def spherical_lng(x,y,z):
    #Omitting the special cases because points will always 
    #be separated for this application
    return (math.atan2(y, x)*rad2deg) # return degrees

def cross_x(x, y, z, i,j,k):
    return ((y*k)-(z*j))
def cross_y(x, y, z, i,j,k):
    return ((z*i)-(x*k))
def cross_z(x, y, z, i,j,k):
    return ((x*j)-(y*i))

def cross_prod(x, y, z, i,j,k):
    return [cross_x(x, y, z, i,j,k), cross_y(x, y, z, i,j,k), cross_z(x, y, z, i,j,k)]    

def midpoint_lat(f0,l0, f1,l1):
    return partial_path_lat(f0,l0, f1,l1,2)

def midpoint_lng(f0,l0, f1,l1):
    #get the x y and z values
    return partial_path_lng(f0,l0, f1,l1,2)

def partial_path_lat(f0,l0, f1,l1, parts):
    #get the x y and z values
    x_0 = cartesian_x(f0,l0)
    y_0 = cartesian_y(f0,l0)
    z_0 = cartesian_z(f0,l0)
    x_1 = cartesian_x(f1,l1)
    y_1 = cartesian_y(f1,l1)
    z_1 = cartesian_z(f1,l1)
   
    x_mid = (x_0+((x_1-x_0)/parts))
    y_mid = (y_0+((y_1-y_0)/parts))
    z_mid = (z_0+((z_1-z_0)/parts))
    print(str(x_mid) + " " + str(y_mid) + " " + str(z_mid))
    return spherical_lat(x_mid, y_mid, z_mid)

def partial_path_lng(f0,l0, f1,l1, parts):
    #get the x y and z values
    x_0 = cartesian_x(f0,l0)
    y_0 = cartesian_y(f0,l0)
    z_0 = cartesian_z(f0,l0)
    x_1 = cartesian_x(f1,l1)
    y_1 = cartesian_y(f1,l1)
    z_1 = cartesian_z(f1,l1)
   
    x_mid = (x_0+((x_1-x_0)/parts))
    y_mid = (y_0+((y_1-y_0)/parts))
    z_mid = (z_0+((z_1-z_0)/parts))
 
    return spherical_lng(x_mid, y_mid, z_mid)


def swept_angle(f0,l0,f1,l1):
    #convert coordinates to Cartesian
    tx_x = cartesian_x(f0,l0)
    tx_y = cartesian_y(f0,l0)
    tx_z = cartesian_z(f0,l0)
    rx_x = cartesian_x(f1,l1)
    rx_y = cartesian_y(f1,l1)
    rx_z = cartesian_z(f1,l1)
    
    g = cross_prod(tx_x, tx_y, tx_z, rx_x, rx_y, rx_z)
    g_mag = math.sqrt(g[0]**2 + g[1]**2 + g[2]**2)
    return math.asin(g_mag)*rad2deg

#Returns lenght of third side of triangle for launch angle
def law_cosines(re, fmax, swangl):
    return math.sqrt(re**2 + (re+fmax)**2 - 2*re*(re+fmax)*math.cos(swangl*deg2rad))

#returns the third angle of the triangle for the launch angle
def law_sines(re, c_side, swangle):
    return math.asin((re*math.sin(swangle*deg2rad))/c_side)*rad2deg

def launch_angle(swangle, sine_angle):
    return ((math.pi - (sine_angle*deg2rad) - (swangle*deg2rad))-(math.pi/2))*rad2deg



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

def load_colors():
    global signal_colors
    signal_colors["1"] = "[" + str(int("96", 16)) + "," + str(int("4b", 16)) + "," +str(int("00", 16)) + "," +str(int("ff", 16)) + "]"
    signal_colors["0"] = "[" + str(int("96", 16)) + "," + str(int("4b", 16)) + "," +str(int("00", 16)) + "," +str(int("ff", 16)) + "]"
    signal_colors["2"] = "[" + str(int("ff", 16)) + "," + str(int("00", 16)) + "," +str(int("00", 16)) + "," +str(int("ff", 16)) + "]"
    signal_colors["3"] = "[" + str(int("ff", 16)) + "," + str(int("a5", 16)) + "," +str(int("00", 16)) + "," +str(int("ff", 16)) + "]"
    signal_colors["4"] = "[" + str(int("ff", 16)) + "," + str(int("ff", 16)) + "," +str(int("00", 16)) + "," +str(int("ff", 16)) + "]"
    signal_colors["5"] = "[" + str(int("00", 16)) + "," + str(int("ff", 16)) + "," +str(int("00", 16)) + "," +str(int("ff", 16)) + "]"
    signal_colors["6"] = "[" + str(int("00", 16)) + "," + str(int("00", 16)) + "," +str(int("ff", 16)) + "," +str(int("ff", 16)) + "]"
    signal_colors["7"] = "[" + str(int("4b", 16)) + "," + str(int("00", 16)) + "," +str(int("82", 16)) + "," +str(int("ff", 16)) + "]"
    signal_colors["8"] = "[" + str(int("7f", 16)) + "," + str(int("00", 16)) + "," +str(int("ff", 16)) + "," +str(int("ff", 16)) + "]"
    signal_colors["9"] = "[" + str(int("ff", 16)) + "," + str(int("ff", 16)) + "," +str(int("ff", 16)) + "," +str(int("ff", 16)) + "]"
    


REQUIRED_COLUMNS = {"tx_lat", "tx_lng", "rx_lat", "rx_lng", "Spotter", "dB"}


@hookimpl
def prepare_connection(conn):
    conn.create_function(
        "hello_world", 0, lambda: "Hello world!"
    )

@hookimpl
def register_output_renderer():
    print("made it into the plugin")
    load_colors()
    return {"extension": "czml", "render": render_czml, "can_render": can_render_atom}

def render_czml(
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
            get_czml(rows),
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

def get_czml(rows):
    from jinja2 import Template
    map_minutes = []
    qso_ends = []
    f2_start = []
    f2_end = []
    f2_height = []
    f2_lat = []
    f2_lng = []
    #not used yet; eventually pass into get_f2m
    f2_station = "EA653"
    mins = time_span(rows)
    print("mins " + str(mins))
    #get the array of minutes ready to go
    map_time = minimum_time(rows)
    for minute in range(mins):
      map_time_str = str(map_time + datetime.timedelta(0,60))
      map_time_str = map_time_str.replace(' ', 'T')
      map_minutes.append(map_time_str)
      map_time = map_time + datetime.timedelta(0,60)
    #Add an end time for each QSO of one minute later (for now)
    f2delta = datetime.timedelta(minutes=5)
    delta = datetime.timedelta(minutes=1)
    for row in rows:
        print(row['timestamp'])
        start_time = datetime.datetime.strptime(row['timestamp'].replace('T',' '), "%Y-%m-%d %H:%M:%S")
        end_time = start_time + delta
        qso_ends.append(datetime.datetime.strftime(end_time, '%Y-%m-%d %H:%M:%S').replace(' ','T'))
        #F2 window
        f2s = datetime.datetime.strptime(row['timestamp'].replace('T',' '), "%Y-%m-%d %H:%M:%S") - f2delta
        f2_start.append(f2s)
        f2e = f2s + f2delta + f2delta
        f2_end.append(f2e)
        f2h = row['f2m']
        print(row['Spotter'] + " f2 height = " + str(f2h) + "km")
        f2_height.append(f2h*1000)
        mid_lng = str(midpoint_lng(float(row['tx_lat']),float(row['tx_lng']),\
                           float(row['rx_lat']),float(row['rx_lng'])))
        mid_lat = str(midpoint_lat(float(row['tx_lat']),float(row['tx_lng']),\
                           float(row['rx_lat']),float(row['rx_lng'])))
        f2_lat.append(mid_lat)
        f2_lng.append(mid_lng)
        
            

    with open('./plugins/templates/qso_map_header.czml') as f:
        #tmpl = Template(f.read())
        tmpl = Environment(loader=FileSystemLoader("./plugins/templates")).from_string(f.read())
        tmpl.globals['line_color'] = line_color
        tmpl.globals['is_qso'] = is_qso
        mit = minimum_time(rows) - delta
        mat = maximum_time(rows) + delta
        mintime = str(mit).replace(' ', 'T')
        #display all the QSOs for a few seconds at the beginning of the maps
        delta = datetime.timedelta(minutes=0.3)
        tmb = mit - delta
        tme = mit + delta
        totmapend=str(tme).replace(' ', 'T')
        totmapbegin=str(tmb).replace(' ', 'T')
        maxtime=str(mat).replace(' ', 'T')
    return(tmpl.render(
        kml_name = 'my first map',
        Rows = rows,
        Map_minutes = map_minutes,
        QSO_ends = qso_ends,
        MinTime = mintime,
        MaxTime = maxtime,
        TotMapEnd = totmapend,
        TotMapBegin = totmapbegin,
        F2Height = f2_height,
        F2Lat = f2_lat,
        F2Lng = f2_lng,
    ))
