from netCDF4 import Dataset
import os, glob
import datetime

gps_epoch = datetime.datetime(1980, 1, 6)

f=open('j_c.csv','w')

f.write("ttj,latitude,longitude,edmaxalt,critfreq\n")
f_count = 0
for filename in glob.glob(os.path.join('./data/', '*.0001_nc')):
     f_count = f_count + 1
     print("f_count = " + str(f_count))
     rootgrp = Dataset(filename, "r")
     gps_seconds = rootgrp.edmaxtime
     datetime_object = gps_epoch + datetime.timedelta(seconds=gps_seconds)
     dt_str = str(datetime_object)
     dt_str = dt_str.replace('-', '/')
     f.write(dt_str + "," + str(rootgrp.edmaxlat) + "," + str(rootgrp.edmaxlon) + "," + str(rootgrp.edmaxalt) + "," + str(rootgrp.critfreq)+"\n")

