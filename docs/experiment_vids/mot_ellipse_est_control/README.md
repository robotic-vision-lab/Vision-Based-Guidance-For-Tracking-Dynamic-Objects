## Multi-Object Tracking


#### experiment MOT + ellipse + dynamic_bb + focal point estimation + control
We start with three targets to track through occlusions. One of them performing lane changing maneuvers while the other two following a straight path. Parameters of the enclosing ellipse is computed and displayed in the tracking window.
For robustness in recovery through prolonged occlusions bounding box sizes are varied according to covariance in estimation.
Controller generated acceleration commands are now applied (not tuned yet). System pipelines seem to be working.
![mot_ellipse_est_control.gif](./gifs/mot_ellipse_est_control.gif)


