wget https://data.cosmic.ucar.edu/gnss-ro/cosmic2/provisional/spaceWeather/level2/2024/$1/ionPrf_prov1_2024_$1.tar.gz
cp ionPrf_prov1_2024_$1.tar.gz data/.
cd data; tar -xvzf ionPrf_prov1_2024_$1.tar.gz; cd ../
python3 ioprf_f2.py
csvs-to-sqlite -dt ttj j_c.csv j_c.db
