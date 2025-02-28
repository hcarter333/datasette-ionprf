from datasette import hookimpl
from datasette.utils.asgi import Response
from jinja2 import Template
from jinja2 import Environment
from jinja2 import FileSystemLoader
import datetime
import math
#from earthmid import midpoint_lng
#from earthmid import midpoint_lat




import math
deg2rad = math.pi/180
rad2deg = 180/math.pi
colormin = 278
colormax = 352
hmF2_colors = 255

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




signal_colors = {"1": "#88004b96",
                 "0": "#66004b96",
                 "2": "#660000ff",
                 "3": "#6600a5ff",
                 "4": "#6600ffff",
                 "5": "#6600ff00",
                 "6": "#66ff0000",
                 "7": "#6682004b",
                 "8": "#66ff007f",
                 "9": "#66ffffff",
                 }

def db_to_s(db):
    test_db = int(db)
    if(test_db > 520):
        return "9"
    if(test_db > 480):
        return "8"
    if(test_db > 440):
        return "7"
    if(test_db > 400):
        return "6"
    if(test_db > 360):
        return "5"
    if(test_db > 320):
        return "4"
    if(test_db > 280):
        return "3"
    if(test_db > 240):
        return "2"
    if(test_db >= 200):
        return "1"
    if(test_db >= 0):
        return "0"
    return "0"

def load_colors():
    global signal_colors
    global hmF2_colors
    signal_colors["1"] = "[" + str(int("96", 16)) + "," + str(int("4b", 16)) + "," +str(int("00", 16)) + "," +str(int(str(hmF2_colors), 16)) + "]"
    signal_colors["0"] = "[" + str(int("00", 16)) + "," + str(int("00", 16)) + "," +str(int("00", 16)) + "," +str(int(str(hmF2_colors), 16)) + "]"
    signal_colors["2"] = "[" + str(int("ff", 16)) + "," + str(int("00", 16)) + "," +str(int("00", 16)) + "," +str(int(str(hmF2_colors), 16)) + "]" #red
    signal_colors["3"] = "[" + str(int("ff", 16)) + "," + str(int("a5", 16)) + "," +str(int("00", 16)) + "," +str(int(str(hmF2_colors), 16)) + "]"
    signal_colors["4"] = "[" + str(int("ff", 16)) + "," + str(int("ff", 16)) + "," +str(int("00", 16)) + "," +str(int(str(hmF2_colors), 16)) + "]"
    signal_colors["5"] = "[" + str(int("00", 16)) + "," + str(int("ff", 16)) + "," +str(int("00", 16)) + "," +str(int(str(hmF2_colors), 16)) + "]" #green
    signal_colors["6"] = "[" + str(int("00", 16)) + "," + str(int("00", 16)) + "," +str(int("ff", 16)) + "," +str(int(str(hmF2_colors), 16)) + "]" #blue
    signal_colors["7"] = "[" + str(int("4b", 16)) + "," + str(int("00", 16)) + "," +str(int("82", 16)) + "," +str(int(str(hmF2_colors), 16)) + "]" #violet
    signal_colors["8"] = "[" + str(int("7f", 16)) + "," + str(int("7f", 16)) + "," +str(int("7f", 16)) + "," +str(int(str(hmF2_colors), 16)) + "]"
    signal_colors["9"] = "[" + str(int("ff", 16)) + "," + str(int("ff", 16)) + "," +str(int("ff", 16)) + "," +str(int(str(hmF2_colors), 16)) + "]"
    


REQUIRED_COLUMNS = {"tx_lat", "tx_lng", "Spotter", "dB", "edmaxalt", "timestamp"}


@hookimpl
def prepare_connection(conn):
    conn.create_function(
        "hello_world", 0, lambda: "Hello world!"
    )

@hookimpl
def register_output_renderer():
    print("made it into the plugin")
    load_colors()
    return {"extension": "iczml", "render": render_czml, "can_render": can_render_atom}

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


def line_color(thmF2_km):
    """
    Returns a string representing an RGBA color for a given hmF2 altitude (in kilometers)
    in the format: "rgba":[R,G,B,255]
    
    The altitude is clamped to the range [150, 500] km, normalized, and mapped to one of 9 colors:
      - Black:   [0, 0, 0, 255]
      - Brown:   [165, 42, 42, 255]
      - Red:     [255, 0, 0, 255]
      - Orange:  [255, 165, 0, 255]
      - Yellow:  [255, 255, 0, 255]
      - Green:   [0, 128, 0, 255]
      - Blue:    [0, 0, 255, 255]
      - Violet:  [238, 130, 238, 255]
      - White:   [255, 255, 255, 255]
    
    Args:
        hmF2_km (float): The hmF2 altitude in kilometers.
        
    Returns:
        str: A string in the format "rgba":[R,G,B,255].
    """
    global hmF2_colors
    # Define altitude limits.
    hmF2_km = float(thmF2_km)
    #We'd like to get as much information out of our color scale 
    #as we can with as little effort as possible. Our user promises 
    #that they will always sort the values for hmF2. The first value will 
    #be the min and the last will be the max
    min_alt_km = colormin
    max_alt_km = colormax
    print("input " + str(hmF2_km) + "colorscale " + str(colormin) + " " + str(colormax))
    # Define the color scale as lists of [R, G, B, A].
    alt_color_scale = [
        [0, 0, 0, hmF2_colors],       # Black
        [165, 42, 42, hmF2_colors],    # Brown
        [255, 0, 0, hmF2_colors],      # Red
        [255, 165, 0, hmF2_colors],     # Orange
        [255, 255, 0, hmF2_colors],     # Yellow
        [0, 128, 0, hmF2_colors],      # Green
        [0, 0, 255, hmF2_colors],      # Blue
        [238, 130, 238, hmF2_colors],     # Violet
        [128, 128, 128, hmF2_colors],    # Grey
        [255, 255, 255, hmF2_colors]     # White
    ]
    n_colors = len(alt_color_scale)

    # Clamp the altitude to the [min_alt_km, max_alt_km] range.
    if hmF2_km < min_alt_km:
        hmF2_km = min_alt_km
    elif hmF2_km > max_alt_km:
        hmF2_km = max_alt_km

    # Normalize the altitude and map it to an index in the color scale.
    range_alt = max_alt_km - min_alt_km
    print("range_alt " + str(range_alt))
    normalized = (hmF2_km - min_alt_km) / range_alt
    if normalized == 1:
        normalized = 0.95
    idx = int(normalized * (n_colors))

    rgba = alt_color_scale[idx]
    # Format the output string exactly as specified.
    return '"rgba":[' + ','.join(str(val) for val in rgba) + ']'













def a_line_color(rst):
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
        time_no_z = row['timestamp'].replace('Z','')
        new_time = datetime.datetime.strptime(time_no_z.replace('T',' '), "%Y-%m-%d %H:%M:%S")
        if new_time < min_time:
            min_time = new_time
    print('found min_time = ' + str(min_time))
    return min_time
    

def maximum_time(rows):
    max_time = datetime.datetime.strptime('1968-02-02 00:00:00', "%Y-%m-%d %H:%M:%S")
    for row in rows:
        time_no_z = row['timestamp'].replace('Z','')
        time_cr = time_no_z.replace('T',' ')        
        new_time = datetime.datetime.strptime(time_cr, "%Y-%m-%d %H:%M:%S")
        if new_time > max_time:
            max_time = new_time
    return max_time    

#Returns the total number of minutes before the first and last QSOs + 5
def time_span(rows):
    #find the largest time
    max_time = datetime.datetime.strptime('1968-02-02 00:00:00', "%Y-%m-%d %H:%M:%S")
    for row in rows:
        print(row['timestamp'])
        time_no_z = row['timestamp'].replace('Z','')
        new_time = datetime.datetime.strptime(time_no_z.replace('T',' '), "%Y-%m-%d %H:%M:%S")
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
    global colormin
    global colormax
    global hmF2_colors
    map_minutes = []
    qso_ends = []
    f2_start = []
    f2_end = []
    f2_height = []
    f2_lat = []
    f2_lng = []
    #set up the color scale
    colormin = float(rows[0]['dB'])
    for row in rows:
        colormax = row['dB']
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
    f2delta = datetime.timedelta(minutes=10)
    delta = datetime.timedelta(minutes=10)
    for row in rows:
        print(row['timestamp'])
        time_no_z = row['timestamp'].replace('Z','')
        time_cr = time_no_z.replace('T',' ')
        start_time = datetime.datetime.strptime(time_cr, "%Y-%m-%d %H:%M:%S")
        end_time = start_time + delta
        qso_ends.append(datetime.datetime.strftime(end_time, '%Y-%m-%d %H:%M:%S').replace(' ','T'))
        #F2 window
        f2s = datetime.datetime.strptime(time_cr, "%Y-%m-%d %H:%M:%S") - f2delta
        f2_start.append(f2s)
        f2e = f2s + f2delta + f2delta
        f2_end.append(f2e)
        f2h = float(row['edmaxalt'])
        print(str(f2h))
        print(str(row['Spotter']) + " f2 height = " + str(f2h) + "km")
        f2_height.append(f2h*1000)
        #mid_lng = str(midpoint_lng(float(row['tx_lat']),float(row['tx_lng']),\
        #                   float(row['rx_lat']),float(row['rx_lng'])))
        #mid_lat = str(midpoint_lat(float(row['tx_lat']),float(row['tx_lng']),\
        #                   float(row['rx_lat']),float(row['rx_lng'])))
        mid_lat = 0
        mid_lng = 0
        f2_lat.append(mid_lat)
        f2_lng.append(mid_lng)
        
    if(rows[0]['hmF2Map'] == 1):
        template_file = './plugins/templates/hmf2_iono_map_header.czml';
        hmF2_colors = 255
    else:
        template_file = './plugins/templates/iono_map_header.czml';
        hmF2_colors = 88
    

    with open(template_file) as f:
        #tmpl = Template(f.read())
        tmpl = Environment(loader=FileSystemLoader("./plugins/templates")).from_string(f.read())
        tmpl.globals['line_color'] = line_color
        tmpl.globals['is_qso'] = is_qso
        mit = minimum_time(rows) - delta
        mat = maximum_time(rows) + delta
        mintime = str(mit).replace(' ', 'T')
        print(str(mit) + " mintime: " + str(mintime))
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
