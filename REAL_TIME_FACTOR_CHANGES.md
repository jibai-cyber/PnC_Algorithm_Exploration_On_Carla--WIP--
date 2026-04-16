# carla_ros_bridge：实时因子（real_time_factor / RTF）相关改动说明

本文档汇总 **carla-ros-bridge** 中为 **同步模式下墙钟节流** 与 **launch 参数传递** 所做的修改

# REF
> https://carla.readthedocs.io/projects/ros-bridge/en/latest/run_ros/#prepare-ros-2-environment  
> https://carla.readthedocs.io/en/latest/adv_synchrony_timestep/#fixed-time-step  
> https://ceti.pages.st.inf.tu-dresden.de/robotics/howtos/SimulationSpeed.html  

---

## 1. `src/carla_ros_bridge/bridge.py`

### 1.1 引入 `time`（约第 16 行）

用于单调时钟与 `sleep`。

```python
import time
```

### 1.2 `main()` 中读取参数（约第 416 行）

启动时把 `real_time_factor` 读入 `parameters` 字典（供日志与 `initialize_bridge`）；默认 **1.0**。设为 **≤0** 可关闭节流（需在 launch 或运行时指定）。

```python
    parameters['real_time_factor'] = carla_bridge.get_param('real_time_factor', 1.0)
```

### 1.3 同步循环内 RTF 节流（约第 250–301 行）

- 每步用 **`get_param`** 读当前 ROS 参数，便于 `ros2 param set` 热更新。
- **`rtf > 0`** 时：本步墙钟耗时 + `sleep(remainder)`，使一步墙钟时间约等于 `fixed_delta_seconds / rtf`。

```python
    def _synchronous_mode_update(self):
        """
        execution loop for synchronous mode
        """
        while not self.shutdown.is_set() and roscomp.ok():
            self.process_run_state()

            try:
                rtf = float(self.get_param(
                    'real_time_factor',
                    self.parameters.get('real_time_factor', 1.0)))
            except (TypeError, ValueError):
                rtf = 0.0
            step_wall_start = time.monotonic() if rtf > 0.0 else None

            if self.parameters['synchronous_mode_wait_for_vehicle_control_command']:
                # fill list of available ego vehicles
                self._expected_ego_vehicle_control_command_ids = []
                with self._expected_ego_vehicle_control_command_ids_lock:
                    for actor_id, actor in self.actor_factory.actors.items():
                        if isinstance(actor, EgoVehicle):
                            self._expected_ego_vehicle_control_command_ids.append(
                                actor_id)

            self.actor_factory.update_available_objects()
            frame = self.carla_world.tick()

            world_snapshot = self.carla_world.get_snapshot()

            self.status_publisher.set_frame(frame)
            self.update_clock(world_snapshot.timestamp)
            self.logdebug("Tick for frame {} returned. Waiting for sensor data...".format(
                frame))
            self._update(frame, world_snapshot.timestamp.elapsed_seconds)
            self.logdebug("Waiting for sensor data finished.")

            if self.parameters['synchronous_mode_wait_for_vehicle_control_command']:
                # wait for all ego vehicles to send a vehicle control command
                if self._expected_ego_vehicle_control_command_ids:
                    if not self._all_vehicle_control_commands_received.wait(CarlaRosBridge.VEHICLE_CONTROL_TIMEOUT):
                        self.logwarn("Timeout ({}s) while waiting for vehicle control commands. "
                                     "Missing command from actor ids {}".format(CarlaRosBridge.VEHICLE_CONTROL_TIMEOUT,
                                                                                self._expected_ego_vehicle_control_command_ids))
                    self._all_vehicle_control_commands_received.clear()

            if step_wall_start is not None:
                sim_dt = float(self.parameters['fixed_delta_seconds'])
                target_wall = sim_dt / rtf
                elapsed = time.monotonic() - step_wall_start
                remainder = target_wall - elapsed
                if remainder > 0.0:
                    time.sleep(remainder)
```

**说明**：仅在 **`synchronous_mode=True` 且本节点负责 `tick()`** 时走此循环；异步 `on_tick` 路径未加墙钟节流。不改变 `fixed_delta_seconds` 的仿真步长定义。

---

## 2. `launch/carla_ros_bridge.launch.py`

### 2.1 Launch 参数声明（约第 42–46 行）

```python
        launch.actions.DeclareLaunchArgument(
            name='real_time_factor',
            default_value='1.0',
            description='Synchronous mode wall-clock pacing: sleep so each step takes fixed_delta_seconds/rtf wall time; 0 disables'
        ),
```

### 2.2 节点参数：单字典 + `real_time_factor`（约第 63–85 行）

将**所有** ROS 参数放在 **一个** `parameters` 字典中，避免 `ros2 launch` 生成多个 `--params-file` 时同名键互相覆盖（例如 `real_time_factor` 被后面的 `0.0` 覆盖）。

```python
        launch_ros.actions.Node(
            package='carla_ros_bridge',
            executable='bridge',
            name='carla_ros_bridge',
            output='screen',
            emulate_tty='True',
            on_exit=launch.actions.Shutdown(),
            parameters=[{
                'use_sim_time': True,
                'host': launch.substitutions.LaunchConfiguration('host'),
                'port': launch.substitutions.LaunchConfiguration('port'),
                'timeout': launch.substitutions.LaunchConfiguration('timeout'),
                'passive': launch.substitutions.LaunchConfiguration('passive'),
                'synchronous_mode': launch.substitutions.LaunchConfiguration('synchronous_mode'),
                'synchronous_mode_wait_for_vehicle_control_command': launch.substitutions.LaunchConfiguration(
                    'synchronous_mode_wait_for_vehicle_control_command'),
                'fixed_delta_seconds': launch.substitutions.LaunchConfiguration('fixed_delta_seconds'),
                'real_time_factor': launch.substitutions.LaunchConfiguration('real_time_factor'),
                'town': launch.substitutions.LaunchConfiguration('town'),
                'register_all_sensors': launch.substitutions.LaunchConfiguration('register_all_sensors'),
                'ego_vehicle_role_name': launch.substitutions.LaunchConfiguration('ego_vehicle_role_name'),
            }]
        )
```

---

## 3. `launch/carla_ros_bridge.launch`（ROS 1 XML）

与 `real_time_factor` 相关的 `arg` / `param`（约第 19–20、42 行）：

```xml
  <!-- real_time_factor: wall-clock pacing in synchronous mode (sim_dt/rtf per step). 0 = disabled. -->
  <arg name='real_time_factor' default='1.0'/>
```

```xml
    <param name="real_time_factor" value="$(arg real_time_factor)"/>
```

---

## 4. `launch/carla_ros_bridge_with_example_ego_vehicle.launch`

透传 `real_time_factor` 到 `carla_ros_bridge.launch`（约第 22、33 行）：

```xml
  <arg name='real_time_factor' default='1.0'/>
```

```xml
    <arg name='real_time_factor' value='$(arg real_time_factor)'/>
```

---

## 5. `launch/carla_ros_bridge_with_example_ego_vehicle.launch.py`

声明默认值并 include 子 launch 时传入（约第 49–67 行）：

```python
        launch.actions.DeclareLaunchArgument(
            name='real_time_factor',
            default_value='1.0'
        ),
        launch.actions.IncludeLaunchDescription(
            launch.launch_description_sources.PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory(
                    'carla_ros_bridge'), 'carla_ros_bridge.launch.py')
            ),
            launch_arguments={
                'host': launch.substitutions.LaunchConfiguration('host'),
                'port': launch.substitutions.LaunchConfiguration('port'),
                'town': launch.substitutions.LaunchConfiguration('town'),
                'timeout': launch.substitutions.LaunchConfiguration('timeout'),
                'passive': launch.substitutions.LaunchConfiguration('passive'),
                'synchronous_mode_wait_for_vehicle_control_command': launch.substitutions.LaunchConfiguration('synchronous_mode_wait_for_vehicle_control_command'),
                'fixed_delta_seconds': launch.substitutions.LaunchConfiguration('fixed_delta_seconds'),
                'real_time_factor': launch.substitutions.LaunchConfiguration('real_time_factor')
            }.items()
        ),
```
