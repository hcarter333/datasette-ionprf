# Creating per QSO Plots of hmF2 and foF2

## Steps
1. python `python glotec.py -outing_glotec https://raw.githubusercontent.com/hcarter333/rm-rbn-history/64e417866d6adeba0ca84cbded9669c5b17370da/qso_update.csv` the csv file is from a qso file from rm-rnb-history
2. Calculate the maximum, (midpoint), F2 height per QSO and corresponding launch angle once `python f2m_launch_once.py --start 2025-08-27T02:34:38 --end 2025-08-27T03:50:38` This assumes the rm_toucans.db database is in place with the same QSOs that are in the slice created in step 2. It also assumes that datasette is running iwwth the slice databases. The command to do this (in a separate terminal) is `python -m datasette glotec_slice.db rm_toucans_slice.db  --plugins-dir=plugins --template-dir plugins/templates --root --static assets:static-files/ --setting sql_time_limit_ms 1700000 --setting max_returned_rows 100000 --cors --crossdb --setting base_url /sam_datasette/ -p 8001`
3. Turn off datasette. Delete the original slices from step 2 and make them again `del rm_toucans_slice.db glotec_slice.db` and then slice again `python glotec.py -slice 2025-08-27T02:34:38Z 2025-08-27T03:50:38Z`
4. Restart datasette with the new slice dbs. You can make the numerical plots using `python qso_graph_batch.py http://127.0.0.1:8001/rm_toucans_slice.json?sql=select%0D%0A++*%0D%0Afrom%0D%0A++rm_rnb_history_pres%0D%0Awhere+db+%3E+100%0D%0A plotsnew3`



**Test Case Catalog** 
| Test name                                 | Purpose                                                |
| ----------------------------------------- | ------------------------------------------------------ |
| `test_same_point_yields_identical`        | verifies the zero-distance branch.                     |
| `test_segment_count_matches_segments_arg` | ensures you get exactly `segments+1` points.           |
| `test_equatorial_quadrant`                | checks a known simple path along the equator.          |
| `test_endpoints_exact_match`              | confirms first/last exactly match inputs.              |
| `test_small_angle_branch`                 | hits the `angle ≈ 0` code path.                        |
| `test_no_redundancy_keeps_all`            | filter should not drop unique-cell points.             |
| `test_basic_redundancy_removal`           | verifies 5°×2.5° cell deduplication.                   |
| `test_boundary_crossing_inclusive`        | tests exact-boundary behavior at multiples of 5°/2.5°. |
| `test_negative_coordinates`               | ensures negative lats/lons floor properly.             |
| `test_combined_pipeline`                  | integration test: no two outputs share the same cell.  | 


##Requires##
numpy 
netCDF4 
datasette-haversine 

