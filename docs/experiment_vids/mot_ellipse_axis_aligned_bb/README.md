## Multi-Object Tracking


#### experiment MOT + ellipse + dynamic_bb + focal point estimation + control + axis aligned bounding box for ellipse

![mot_ellipse_axis_aligned_bb.gif](./gifs/mot_ellipse_axis_aligned_bb.gif)

We start with three targets to track through occlusions. One of them performing lane changing maneuvers while the other two following a straight path. Parameters of the enclosing ellipse is computed and displayed in the tracking window.
For robustness in recovery through prolonged occlusions bounding box sizes are varied according to covariance in estimation.
Controller generated acceleration commands are now applied (not tuned yet). Axis aligned box is produced for developing simple rules for altitude control


