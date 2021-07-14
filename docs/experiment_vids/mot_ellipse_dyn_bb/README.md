## Multi-Object Tracking


#### experiment MOT + ellipse + dynamic_bb
Controller is turned off. We start with three targets to track through occlusions. One of them performing lane changing maneuvers while the other two following a straight path. Parameters of the enclosing ellipse is computed and displayed in the tracking window.
For robustness in recovery through prolonged occlusions bounding box sizes are varied according to covariance in estimation.

![mot_dyn_bb.gif](./gifs/mot_dyn_bb.gif)


