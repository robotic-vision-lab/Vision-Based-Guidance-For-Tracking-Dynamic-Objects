## Multi-Object Tracking


#### experiment MOT Lane Changing Variant

![mot_lane_change.gif](./gifs/mot_lane_change.gif)

We start with three targets to track through occlusions. One of them performing lane changing maneuvers while the other two following a straight path. Parameters of the enclosing ellipse is computed and displayed in the tracking window.
For robustness in recovery through prolonged occlusions bounding box sizes are varied according to covariance in estimation.
Controller generated acceleration commands a_lat, a_long and a_z are applied.


