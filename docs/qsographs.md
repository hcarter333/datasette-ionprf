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
