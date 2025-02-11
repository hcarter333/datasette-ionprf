from datasette import hookimpl
from haversine import haversine
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

def gis_partial_path_lat_sql(lat1, lon1, lat2, lon2, partial_dist):
    #get the entire length of the path
    full_path = haversine((float(lat1), float(lon1)), (float(lat2), float(lon2)), unit='m')

    #calculate the portion of the path needed for the partial path
    frac_path = full_path/partial_dist
    
    #use earth_mid to find hte latitude of the endpoint
    return partial_path_lat(float(lat1),float(lon1), float(lat2) ,float(lon2), frac_path)
    
def gis_partial_path_lng_sql(lat1, lon1, lat2, lon2, partial_dist):
    #get the entire length of the path
    full_path = haversine((float(lat1), float(lon1)), (float(lat2), float(lon2)), unit='m')

    #calculate the portion of the path needed for the partial path
    frac_path = full_path/partial_dist
    
    #use earth_mid to find hte latitude of the endpoint
    return partial_path_lng(float(lat1),float(lon1), float(lat2) ,float(lon2), frac_path)


@hookimpl
def prepare_connection(conn):
    conn.create_function("gis_partial_path_lng", 5, gis_partial_path_lng_sql)
    conn.create_function("gis_partial_path_lat", 5, gis_partial_path_lat_sql)
