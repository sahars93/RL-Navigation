# RL-omni

## Installation


Follow the Isaac Sim [documentation](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/install_basic.html) to install the latest Isaac Sim release.


To install `omniisaacgymenvs`, first clone this repository:

```bash
git clone https://github.com/sahars93/RL-omni.git
```

Set a `PYTHON_PATH` variable in the terminal that links to the python executable, 

```
alias PYTHON_PATH=~/.local/share/ov/pkg/isaac_sim-*/python.sh
```

Install `omniisaacgymenvs` as a python module for `PYTHON_PATH`. Change directory to root of this repo and run:

```bash
PYTHON_PATH -m pip install -e .
```

## Training

cd to omniisaacgymenvs

```bash
PYTHON_PATH scripts/rlgames_train.py task=Jetbot pipeline=cpu
```

## Exporting the neural network

Export the .pth model to onnx format

```bash
PYTHON_PATH scripts/rlgames_onnx_normalized.py task=Jetbot test=True checkpoint=CHECKPOINT_PATH pipeline=cpu
```

## Test in Gazebo

Install gazebo Turtlebot3 simulation package

```bash
cd ~/catkin_ws/src/
git clone https://github.com/ROBOTIS-GIT/turtlebot3_simulations.git
cd ~/catkin_ws && catkin_make
```
Change the number of 2d lidar samples (to 21) and the min range (to 0.1) and the max range (to 1.0):

```bash
udo nano /opt/ros/noetic/share/turtlebot3_description/urdf/turtlebot3_waffle.gazebo.xacro
```

Launch the TurtleBot3 World


```bash
export TURTLEBOT3_MODEL=burger
roslaunch turtlebot3_gazebo turtlebot3_world.launch
```

To test the model in gazebo run the ROS NODE.

```bash
python3 ROS/gazebo_test.py
```
