# Intro
See updated_version.txt  

# Ref
> https://github.com/gezp/carla_ros/releases/  
> https://carla.readthedocs.io/projects/ros-bridge/en/latest/  
> https://autowarefoundation.github.io/autoware-documentation/main/tutorials/  
> https://github.com/fzi-forschungszentrum-informatik/Lanelet2/  
> https://github.com/ApolloAuto/apollo/  

---

# In terminal with ros2
```bash
export ROS_DOMAIN_ID=200
```

# Terminal 1, open carla simulator
```bash
cd carla0914/
./CarlaUE4.sh 
```

# Terminal 2, load carla-ros-bridge
```bash
source ~/carla-ros-bridge/catkin_ws/install/setup.bash
ros2 launch carla_ros_bridge carla_ros_bridge.launch.py synchronous_mode:=True town:=Town01
```

# Terminal 3, generate a car
```bash
source ~/carla-ros-bridge/catkin_ws/install/setup.bash
```
## Config of carla is modifiable in _objects.json_
```bash
ros2 launch carla_spawn_objects carla_example_ego_vehicle.launch.py spawn_sensors_only:=False objects_definition_file:=/home/usr/ws/src/vehicle_ctrl/config/objects.json 
```

# Terminal 4, load map
```bash
source ~/ws/install/setup.bash 
ros2 run map_load map_control_node 
```

# Terminal 5, remap goal
```bash
source ~/ws/install/setup.bash 
ros2 run vehicle_ctrl remap_goal
```

# Terminal 6, open rviz
```bash
rviz2 -d src/vehicle_ctrl/rviz2/carla_map_spawn_anywherev2.rviz 
```

# Terminal 7, open control/smooth/perception node
```bash
source ~/ws/install/setup.bash 
source ~/carla-ros-bridge/catkin_ws/install/setup.bash
ros2 launch vehicle_ctrl full_system.launch.py
```

# Terminal 8, open the plotter
```bash
source ~/ws/install/setup.bash 
ros2 run vehicle_ctrl vehicle_plotter
```