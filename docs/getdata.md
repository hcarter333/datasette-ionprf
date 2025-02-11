wget https://data.cosmic.ucar.edu/gnss-ro/cosmic2/provisional/spaceWeather/level2/2024/158/ionPrf_prov1_2024_158.tar.gz
ioprf_f2.py
csvs-to-sqlite -dt ttj j_c.csv j_c.db
