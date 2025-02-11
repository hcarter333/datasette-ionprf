import requests
import datetime
import urllib3

#returns the first hmF2 from DADBGetValues for the time window specified
def get_f2m(start_time, end_time, station = "EA653"):
#def get_f2m():
    hmF2 = "5"
    #station = "EA653"
    ptargstation = "PA836"
    #2023/02/08 01:34:00
    start_date_win = start_time.strftime("%Y.%m.%d %H:%M:%S")
    start_date_win = start_date_win.replace(" ", "%20")
    #print(start_date_win)
    end_date_win = end_time.strftime("%Y.%m.%d %H:%M:%S")
    end_date_win = end_date_win.replace(" ", "%20")
    #print("start_date_win " + start_date_win)
    #print("end_date_win " + end_date_win)
    urllib3.disable_warnings()
    iono_url = "https://lgdc.uml.edu/common/DIDBGetValues?ursiCode="+station+"&charName=hF2,hmF2&fromDate=" + start_date_win + \
                        "&toDate=" + end_date_win
    print("iono_url = " + iono_url)
    #                                                                                                        2023.02.08%2002:50:00
    iodata=requests.get(iono_url, verify=False)
    #print(iodata.text)
    iono_data = iodata.text
    iono_lines = iono_data.split("\n")
    data_found = 0
    for data_line in iono_lines:
        #print(data_line)
        if(data_found == 1):
            #We've arrived at the data
            #Split on space and print the max F2
            data_fields = data_line.split()
            print(data_fields)
            hmF2 = data_fields[4]
            #print("hmF2 " + data_fields[4])
            break
    #we're going to use the closest time to the beginning of the range for now
        if(data_line.find("#Time") != -1):
            data_found = 1

    return_h = 0.0
    try:
        return_h = float(hmF2)
        #print("freq " + str(fields[3]))
        return return_h
    except Exception as error:
        return_h = float(data_fields[2])
        return return_h

    return return_h