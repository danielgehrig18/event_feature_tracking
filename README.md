# Event Feature Tracking
This repo contains MATLAB implementations of the event-based feature tracking methods described in [Event-based Feature Tracking with Probabilistic Data Association](https://doi.org/10.1109/ICRA.2017.7989517) and [Event-based Visual Inertial Odometry](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhu_Event-Based_Visual_Inertial_CVPR_2017_paper.pdf).

## Data
We provide functions in the ```data``` folder to convert event data from ROS bag to MATLAB mat files. Currently, data from the [Event Camera Dataset](http://rpg.ifi.uzh.ch/davis_data.html) and the [Multi Vehicle Stereo Event Camera dataset](https://daniilidis-group.github.io/mvsec/) is tested. The [```extract_ros_data.m```](https://github.com/daniilidis-group/EventFeatureTracking/blob/master/data/extract_ros_data.m) and either the [```generate_undistort_map_radtan.m```](https://github.com/daniilidis-group/EventFeatureTracking/blob/master/data/generate_undistort_map_radtan.m) or [```generate_undistort_map_equi.py```](https://github.com/daniilidis-group/EventFeatureTracking/blob/master/data/generate_undistort_map_equi.py) functions are necessary to generate the data and undistort map mat files. You must select the appropriate undistort map function depending on whether the camera calibration uses the radtan distortion model (e.g. the Event Camera Dataset) or the equidistant distortion model (e.g. MVSEC). Note that you will need the [matlab_rosbag package](https://github.com/bcharrow/matlab_rosbag), which has pre-compiled releases [here](https://github.com/bcharrow/matlab_rosbag/releases). Assuming you have the ```boxes_6dof.bag``` file in the data folder, the extraction code can be run as follows:
~~~~
load_folder = 'data';
save_folder = 'data';
extract_ros_data('boxes_6dof', load_folder, save_folder)
~~~~
For the radtan distortion model, the undistort maps are generated in MATLAB as follows:
~~~~
load data/boxes_6dof.mat
generate_undistort_map_radtan(cinfo, 'boxes_6dof')
~~~~
For the equidistant distortion model, the undistort maps are generated from the terminal as follows:
~~~~
python generate_undistort_map_radtan.py --camchain data/camchain-imucam-indoor_flying1.yaml
~~~~
Note that the undistort_map can be shared across sequences with the same camera intrinsics.

## Running the code
The code can be run from the script [```main.m```](https://github.com/daniilidis-group/EventFeatureTracking/blob/master/EventFeatureTracking/main.m). Before you do so, you must set a few parameters, which are stored in the function [```get_params.m```](https://github.com/daniilidis-group/EventFeatureTracking/blob/master/EventFeatureTracking/get_params.m). At a minimum, you will need to set the path to the generated data mat ([```params.data_path```](https://github.com/daniilidis-group/EventFeatureTracking/blob/d9867d973d74bb4da12227d86c26ef443ef42026/EventFeatureTracking/get_params.m#L27)), as well as the generated undistort_map mat ([```params.undistort_data_path```](https://github.com/daniilidis-group/EventFeatureTracking/blob/d9867d973d74bb4da12227d86c26ef443ef42026/EventFeatureTracking/get_params.m#L29)). This function also provides many other parameters, such as start and end times in the sequence, number of features and many more. For optimal performance, the main tracking loop can be run in parallel by setting [```params.do_parallel```](https://github.com/daniilidis-group/EventFeatureTracking/blob/d9867d973d74bb4da12227d86c26ef443ef42026/EventFeatureTracking/get_params.m#L48) to ```true```. 

If you are using the IMU related features (for the EM2 affine warp or two point RANSAC), you will need the [Robotics System Toolbox](https://www.mathworks.com/products/robotics.html), or to define your own functions for [```quat2rotm```](https://www.mathworks.com/help/robotics/ref/quat2rotm.html) and [```rotm2quat```](https://www.mathworks.com/help/robotics/ref/rotm2quat.html), which convert between a quaternion (w,x,y,z) and a 3x3 rotation matrix.

The feature points and their corresponding IDs are stored as the variables [```valid_feature_positions```](https://github.com/daniilidis-group/EventFeatureTracking/blob/d9867d973d74bb4da12227d86c26ef443ef42026/EventFeatureTracking/main.m#L346) and [```valid_ids```](https://github.com/daniilidis-group/EventFeatureTracking/blob/d9867d973d74bb4da12227d86c26ef443ef42026/EventFeatureTracking/main.m#L347), and are updated at the end of each iteration in [```main.m```](https://github.com/daniilidis-group/EventFeatureTracking/blob/master/EventFeatureTracking/main.m). If [```params.headless```](https://github.com/daniilidis-group/EventFeatureTracking/blob/d9867d973d74bb4da12227d86c26ef443ef42026/EventFeatureTracking/get_params.m#L52) is set to ```false```, the features will be plotted on top of the integrated event image in Figure 1.

Note that the label "EM1 flow" refers to the optical flow estimation step (Section IV.B in Event-based Feature Tracking with Probabilistic Data Association), and the label "EM2 affine" refers to the feature alignment step (Section IV.C in Event-based Feature Tracking with Probabilistic Data Association).

### Debugging
To debug the underlying EM algorithms, set [```params.debug```](https://github.com/daniilidis-group/EventFeatureTracking/blob/d9867d973d74bb4da12227d86c26ef443ef42026/EventFeatureTracking/get_params.m#L41) to ```true```. This will plot the optimization process at each step, and may help give you an intuition behind what is being minimized. Note that setting this to ```true``` will automatically set [```params.do_parallel```](https://github.com/daniilidis-group/EventFeatureTracking/blob/d9867d973d74bb4da12227d86c26ef443ef42026/EventFeatureTracking/get_params.m#L48) to ```false```, as plotting doesn't work properly inside a parfor loop.

### Benchmarks
For comparisons and benchmarks, the method proposed in "Event-based Feature Tracking with Probabilistic Data Association" can be run by setting [```params.use_em1_template```](https://github.com/daniilidis-group/EventFeatureTracking/blob/d9867d973d74bb4da12227d86c26ef443ef42026/EventFeatureTracking/get_params.m#L73), [```params.use_imu```](https://github.com/daniilidis-group/EventFeatureTracking/blob/d9867d973d74bb4da12227d86c26ef443ef42026/EventFeatureTracking/get_params.m#L54) and [```params.do_ransac```](https://github.com/daniilidis-group/EventFeatureTracking/blob/d9867d973d74bb4da12227d86c26ef443ef42026/EventFeatureTracking/get_params.m#L46) to ```false```. Setting these parameters to ```true``` will run the full method proposed in "Event-based Visual Inertial Odometry".

## Citations
If you use this code in an academic publication, please cite the following works:

Alex Zihao Zhu, Nikolay Atanasov and Kostas Daniilidis. "Event-based Feature Tracking with Probabilistic Data Association", IEEE International Conference on Robotics and Automation (ICRA), 2017.

Alex Zihao Zhu, Nikolay Atanasov, and Kostas Daniilidis. "Event-Based Visual Inertial Odometry." IEEE International Conference on Computer Vision and Pattern Recognition (CVPR), 2017.

This technology is the subject of a pending patent application: PCT/US2018/018196
