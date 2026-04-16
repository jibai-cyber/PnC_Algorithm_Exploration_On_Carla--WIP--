cmake_minimum_required(VERSION 3.8)
project(map_load)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(derived_object_msgs REQUIRED)

# Generate custom messages
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/FrenetPath.msg"
  "msg/SLBoundary.msg"
  "msg/SLBoundaryArray.msg"
  "msg/PathBoundary.msg"
  "msg/PlanningPathPoint.msg"
  "msg/PlanningSpeedPoint.msg"
  "msg/PlanningTrajectoryPoint.msg"
  "msg/LocalPlanningPath.msg"
  "msg/PlanningSpeedProfile.msg"
  "msg/EgoPlanningTrajectory.msg"
  "msg/STObstacleRegion.msg"
  "msg/STGraph.msg"
  "msg/PlanningObstacle.msg"
  "msg/PlanningObstacleArray.msg"
  DEPENDENCIES std_msgs derived_object_msgs
)

# Install Python modules to the same location as message bindings
# rosidl_generate_interfaces installs to local/lib/python3.10/dist-packages,
# so we install Python modules there too to ensure they can be found together
install(DIRECTORY
  ${PROJECT_NAME}/
  DESTINATION local/lib/python3.10/dist-packages/${PROJECT_NAME}
  FILES_MATCHING
  PATTERN "*.py"
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)

# Install resource files (map files, etc.)
install(DIRECTORY
  resource/
  DESTINATION share/${PROJECT_NAME}/resource
  FILES_MATCHING
  PATTERN "*.osm"
  PATTERN "*.pcd"
)

# Install package.xml and resource marker
install(FILES
  package.xml
  DESTINATION share/${PROJECT_NAME}
)

install(FILES
  resource/${PROJECT_NAME}
  DESTINATION share/ament_index/resource_index/packages
)

# Install entry point scripts
find_program(PYTHON3 "python3" REQUIRED)

# map_control_node
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/map_control_node"
  CONTENT "#!/usr/bin/env python3\nimport sys\nfrom map_load.map_control import main\nif __name__ == '__main__':\n    sys.exit(main())\n"
)
install(PROGRAMS "${CMAKE_CURRENT_BINARY_DIR}/map_control_node"
  DESTINATION lib/${PROJECT_NAME}
)

# path_smoother_node
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/path_smoother_node"
  CONTENT "#!/usr/bin/env python3\nimport sys\nfrom map_load.path_smoother import main\nif __name__ == '__main__':\n    sys.exit(main())\n"
)
install(PROGRAMS "${CMAKE_CURRENT_BINARY_DIR}/path_smoother_node"
  DESTINATION lib/${PROJECT_NAME}
)

# vehicle_perception
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/vehicle_perception"
  CONTENT "#!/usr/bin/env python3\nimport sys\nfrom map_load.vehicle_perception import main\nif __name__ == '__main__':\n    sys.exit(main())\n"
)
install(PROGRAMS "${CMAKE_CURRENT_BINARY_DIR}/vehicle_perception"
  DESTINATION lib/${PROJECT_NAME}
)

# speed_planner
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/speed_planner"
  CONTENT "#!/usr/bin/env python3\nimport sys\nfrom map_load.speed_planner import main\nif __name__ == '__main__':\n    sys.exit(main())\n"
)
install(PROGRAMS "${CMAKE_CURRENT_BINARY_DIR}/speed_planner"
  DESTINATION lib/${PROJECT_NAME}
)

# file(GENERATE) creates files without the executable bit; --symlink-install then
# points lib/pkg/exe -> build/.../exe, and ros2 run skips non-X_OK paths ("No executable found").
add_custom_target(map_load_entrypoints_executable ALL
  COMMAND chmod a+x
          "${CMAKE_CURRENT_BINARY_DIR}/map_control_node"
          "${CMAKE_CURRENT_BINARY_DIR}/path_smoother_node"
          "${CMAKE_CURRENT_BINARY_DIR}/vehicle_perception"
          "${CMAKE_CURRENT_BINARY_DIR}/speed_planner"
  COMMENT "chmod +x map_load ros2 run entry points")

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
