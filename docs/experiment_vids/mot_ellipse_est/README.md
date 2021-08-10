## Multi-Object Tracking


#### experiment MOT + ellipse + dynamic_bb + focal point estimations

![mot_ellipse_est.gif](./gifs/mot_ellipse_est.gif)

Controller is off. We start with three targets to track through occlusions. One of them performing lane changing maneuvers while the other two following a straight path. Parameters of the enclosing ellipse is computed (transformed form inertial world frame (m) to non-inertial camera frame (px)) and displayed in the tracking window.
For robustness in recovery through prolonged occlusions bounding box sizes are varied according to covariance in estimation.
Additionally, ellipse parameters that are originally computed in the world inertial frame and focal points obtained to filter using EKF.



