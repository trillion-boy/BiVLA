# False discovery rate check over the 110 table cells

Source: paired_results_all.csv (286 settings, `in_table` = 1 on 110). p is the exact two-sided McNemar test on the flipped episodes; the table cell is the best of two or three settings, so its p is post-selection. Benjamini-Hochberg at q = 0.05 over the 110 table cells.

## Summary

| quantity | paper (Sec. IV-A / IV-B) | reproduced |
|---|---|---|
| table cells | 110 | 110 |
| cells that lower success with p < 0.05 | 22 | 22 |
| cells that raise success with p < 0.05 | 4 | 4 |
| significant drops passing BH (q = 0.05, m = 110) | 16 | 16 |
| significant gains passing BH | 0 | 0 |
| BH passes in total | 16 | 16 |
| settings (of 286) that raise success with p < 0.05 | 6 | 6 |
| settings (of 286) that lower success with p < 0.05 | (not stated) | 77 |
| expected chance passes over 286 settings (sum of exact test sizes at 0.05) | about 8 (4 per direction) | 8.0 (4.0 per direction) |
| expected chance passes over 110 cells | (comment: 3.1) | 3.1 |
| guarded reuse settings with a significant drop (of 66) | 0 | 0 |

Reproduced: **yes**. The BH threshold for rank k is k x 0.05 / 110; the largest rank whose p is at or below its threshold is 16, so the p cut-off is 0.0061.

## The 22 significant table drops

| backbone | env | setting | delta (pts) | p | BH q=0.05 |
|---|---|---|---|---|---|
| CogACT | WidowX | action_repeat2 | -38.0 | 0.0000 | pass |
| CronusVLA | WidowX | action_repeat2 | -30.5 | 0.0000 | pass |
| MiniVLA | WidowX | depth_pruning1 | -17.5 | 0.0000 | pass |
| UniVLA | WidowX | action_repeat2 | -75.0 | 0.0000 | pass |
| UniVLA | LIBERO Long | action_repeat2 | -59.0 | 0.0000 | pass |
| UniVLA | LIBERO Goal | action_repeat2 | -58.0 | 0.0000 | pass |
| UniVLA | LIBERO Object | action_repeat2 | -75.0 | 0.0000 | pass |
| UniVLA | LIBERO Spatial | action_repeat2 | -68.0 | 0.0000 | pass |
| SmolVLA | LIBERO Long | depth_pruning1 | -33.0 | 0.0000 | pass |
| SmolVLA | LIBERO Object | depth_pruning1 | -21.0 | 0.0001 | pass |
| SmolVLA | LIBERO Long | fixed_foveation_keep50 | -21.0 | 0.0003 | pass |
| OpenVLA | WidowX | action_repeat2 | -15.5 | 0.0005 | pass |
| OpenVLA | Fractal | fixed_foveation_keep50 | -11.2 | 0.0013 | pass |
| SmolVLA | LIBERO Goal | depth_pruning1 | -18.0 | 0.0014 | pass |
| SmolVLA | LIBERO Object | fixed_foveation_keep50 | -13.0 | 0.0023 | pass |
| OpenVLA | WidowX | fixed_foveation_keep50 | -13.5 | 0.0061 | pass |
| UniVLA | WidowX | fixed_foveation_keep20 | -8.5 | 0.0095 | fail |
| OpenVLA | LIBERO Spatial | depth_pruning1 | -16.0 | 0.0113 | fail |
| SmolVLA | LIBERO Goal | fixed_foveation_keep50 | -13.0 | 0.0146 | fail |
| SpatialVLA | WidowX | depth_pruning1 | -6.5 | 0.0241 | fail |
| SmolVLA | LIBERO Spatial | depth_pruning1 | -14.0 | 0.0336 | fail |
| OpenVLA | LIBERO Long | fixed_foveation_keep50 | -12.0 | 0.0357 | fail |

## The 4 significant table gains

| backbone | env | setting | delta (pts) | p | BH q=0.05 |
|---|---|---|---|---|---|
| CogACT | WidowX | depth_pruning2 | +9.5 | 0.0094 | fail |
| OpenVLA | Fractal | temporal_fusion_task_aware | +6.8 | 0.0115 | fail |
| CogACT | WidowX | temporal_fusion_motion_entropy | +8.0 | 0.0195 | fail |
| UniVLA | LIBERO Goal | fixed_foveation_keep50 | +7.0 | 0.0391 | fail |

## Significant gains over all 286 settings

| backbone | env | setting | delta (pts) | p | in table |
|---|---|---|---|---|---|
| CogACT | WidowX | depth_pruning2 | +9.5 | 0.0094 | yes |
| OpenVLA | Fractal | temporal_fusion_task_aware | +6.8 | 0.0115 | yes |
| CogACT | WidowX | temporal_fusion_motion_entropy | +8.0 | 0.0195 | yes |
| CogACT | WidowX | temporal_fusion_task_aware | +8.0 | 0.0195 |  |
| CogACT | WidowX | depth_pruning1 | +7.0 | 0.0385 |  |
| UniVLA | LIBERO Goal | fixed_foveation_keep50 | +7.0 | 0.0391 | yes |

## Consistency of pvalues.csv (project_page) with paired_results_all.csv

All 286 rows agree on n, flips, delta, p (to 1e-4) and the table flag.
