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



#half_lat = midpoint_lat(37.72286,-122.42299, 41.0625,-112.0417)
#half_lng = midpoint_lng(37.72286,-122.42299, 41.0625,-112.0417)

#print(str(half_lat) + "," + str(half_lng))