# Intro
See updated_version.txt
> https://github.com/gezp/carla_ros/releases/
> https://carla.readthedocs.io/projects/ros-bridge/en/latest/

# in terminal with ros2
```bash
export ROS_DOMAIN_ID=200
```

# terminal 1, open carla simulator
```bash
cd carla0914/
./CarlaUE4.sh 
```

# terminal 2, activate conda env, load carla-ros-bridge
```bash
source ~/carla-ros-bridge/catkin_ws/install/setup.bash
ros2 launch carla_ros_bridge carla_ros_bridge.launch.py synchronous_mode:=True town:=Town01
```

# terminal 3, generate a car
```bash
source ~/carla-ros-bridge/catkin_ws/install/setup.bash
```
## objects.json用于定义在 CARLA 仿真环境中要生成的车辆和传感器,路径需要根据自己的目录改
```bash
ros2 launch carla_spawn_objects carla_example_ego_vehicle.launch.py spawn_sensors_only:=False objects_definition_file:=/home/usr/ws/src/vehicle_ctrl/config/objects.json 
```

# terminal 4, load map
```bash
source ~/ws/install/setup.bash 
ros2 run map_load map_control_node 
```

# terminal 5, remap goal
```bash
source ~/ws/install/setup.bash 
ros2 run vehicle_ctrl remap_goal
```

# terminal 6, open rviz
```bash
rviz2 -d src/vehicle_ctrl/rviz2/carla_map_spawn_anywherev2.rviz 
```

# terminal 7, open control node
```bash
source ~/ws/install/setup.bash 
source ~/carla-ros-bridge/catkin_ws/setup.bash
ros2 run vehicle_ctrl simple_ctrl 
```

# terminal 8, open plotter
```bash
source ~/ws/install/setup.bash 
ros2 run vehicle_ctrl vehicle_plotter
```
