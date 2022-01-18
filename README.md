## Sensor Placement Simulation

The "gazebo_setup" branch comprises of "vehicle_sim" package. It contains the simulation of proposed sensor placements in GAZEBO. 
Execute following commands to compile the package:


```
git clone https://komalb_@bitbucket.org/alive_iiitd/sensoroncar_simulation.git
cd sensoroncar_simulation
git checkout gazebo_setup
rosdep install --from-paths src/vehicle_sim -y
sudo apt-get install ros-melodic-gazebo-ros-pkgs ros-melodic-gazebo-ros-control
catkin_make
source /devel/setup.bash
rosrun vehicle_sim_launcher setup.sh
```

#### To visualise the Lidar pointcloud with gpu:

```
roslaunch vehicle_sim_launcher world_test.launch gpu:=true rvizconfig:=/home/user/sensoroncar_simulation/src/vehicle_sim/urdf.rviz
```

#### To visualise the camera topics with gpu:

```
roslaunch vehicle_sim_launcher world_test.launch gpu:=true rvizconfig:=/home/user/sensoroncar_simulation/src/vehicle_sim/urdf_camera.rviz
```


## To Run Carla Script

It contains the lidar and camera setup simulation as of November 2020.

#### Install pygame, numpy and other dependencies for Carla visualisation

```
sudo apt-get install -y libomp5 libomp-dev freeglut3 freeglut3-dev libxi-dev libxmu-dev mesa-utils 
sudo apt-get install -y libgl1-mesa-glx pkg-config libxau-dev libxdmcp-dev libxcb1-dev libxext-dev libx11-dev
sudo apt-get install vulkan-utils
pip3 install --user pygame numpy
```

#### Install Carla to run with pygame

```
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 1AF1527DE64CB8D9
sudo add-apt-repository "deb [arch=amd64] http://dist.carla.org/carla $(lsb_release -sc) main"
sudo apt-get update 
sudo apt-get install carla-simulator 
cd /opt/carla-simulator 
./ImportAssets.sh
```

#### Run Carla Script

```
./CarlaUE4.sh
cd carla_script
python3 carla_car_setup_11-20.py
```

For more information:
https://carla.readthedocs.io/en/0.9.10/start_quickstart/
