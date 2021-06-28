## Vision-Based Object Tracking for Unmanned Aircraft Systems

[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/keras-team/keras/blob/master/LICENSE)

![occlusion_handling.gif](./docs/experiment_vids/final_experiments/gifs/occ.gif)

### Overview
One key application of computer vision is performing object tracking with robots through the use of visual information. This repository consists of the code base and documentation for a vision-based object tracking system. Specifically, we implement experiments for performing the diagnosis and analysis of visual tracking techniques under occlusions along with UAS guidance based on a rendezvous cone approach.

### Purpose
This software contains computer vision algorithms that may be used in various combinations, pipelines, or standalone depending upon the complexity and/or requirement of the task. The repository was created to contain simulation experiments for our 2021 ICUAS paper entitled "[Vision-Based Guidance for Tracking Dynamic Objects](https://arxiv.org/abs/2104.09301)." If you find this project useful, then please consider citing our work:
```
@inproceedings{karmokar2021vision,
  title={Vision-Based Guidance for Tracking Dynamic Objects},
  author={Karmokar, Pritam and Dhal, Kashish and Beksi, William J and Chakravarthy, Animesh},
  booktitle={Proceedings of the International Conference on Unmanned Aircraft Systems (ICUAS)},
  pages={1106--1115},
  year={2021}
}
```

### Installation
To run the experiments within this repository, `opencv`, `numpy`, and `pygame` need to be installed along with their dependencies. The `requirements.txt` file (generated by `pip freeze`) may be used as follows. Navigate into the downloaded source folder where `requirements.txt` is located. Then, run the following
```python
pip install -r requirements.txt
```

### Usage
From the source folder, navigate into the experiments folder
```bash
cd .\vbot\experiments
```
To run the occlusion handling experiment, run the following
```python
python -m exp_occ
```
To run the lane changing experiment, run the following
```python
python -m exp_lc
```
To run the squircle following experiment, run the following
```python
python -m exp_sf
```

### Running Experiments
The process of running experiments has the following steps.

1. Simulator window pops up.
2. User inputs bounding boxing around car (with some extra room).
3. Users hits `space` to start experiment. 
4. Tracker window appears displaying tracking results.
5. To stop the experiment, user selects the simulator window and hits `space`, closes the window.


<!-- 

### Papers

#### Optical Flow
* **1981**
    * [Determining Optical Flow](http://image.diku.dk/imagecanon/material/HornSchunckOptical_Flow.pdf)
    * [An iterative image registration technique with an application to stereo vision](https://cecas.clemson.edu/~stb/klt/lucas_bruce_d_1981_1.pdf)
* **1993**
    * [Good Features to Track](http://www.ai.mit.edu/courses/6.891/handouts/shi94good.pdf) -->


### License

[MIT](https://github.com/robotic-vision-lab/Vision-Based-Ojbect-Tracking/blob/master/LICENSE)

