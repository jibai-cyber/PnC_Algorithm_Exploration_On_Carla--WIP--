cmake_minimum_required(VERSION 3.8)
project(planning)

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
  "msg/STObstacleStamp.msg"
  "msg/STGraph.msg"
  "msg/PlanningObstacle.msg"
  "msg/PlanningObstacleArray.msg"
  DEPENDENCIES std_msgs derived_object_msgs
)

# Install Python modules to the same location as message bindings
# rosidl_generate_interfaces installs to local/lib/python3.10/dist-packages,
# so we install Python modules there too to ensure they can be found together
install(DIRECTORY
  modified_EM_planner/
  DESTINATION local/lib/python3.10/dist-packages/modified_EM_planner
  FILES_MATCHING
  PATTERN "*.py"
  PATTERN "__pycache__" EXCLUDE
  PATTERN "*.pyc" EXCLUDE
)

# Install config files
install(DIRECTORY
  config/
  DESTINATION share/${PROJECT_NAME}/config
  FILES_MATCHING
  PATTERN "*.json"
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
  resource/planning
  DESTINATION share/ament_index/resource_index/packages
)

# Install entry point scripts
find_program(PYTHON3 "python3" REQUIRED)

# global_path_planner_node
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/global_path_planner_node"
  CONTENT "#!/usr/bin/env python3\nimport sys\nfrom modified_EM_planner.global_path_planner import main\nif __name__ == '__main__':\n    sys.exit(main())\n"
)
install(PROGRAMS "${CMAKE_CURRENT_BINARY_DIR}/global_path_planner_node"
  DESTINATION lib/${PROJECT_NAME}
)

# ref_line_smoother_node
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/ref_line_smoother_node"
  CONTENT "#!/usr/bin/env python3\nimport sys\nfrom modified_EM_planner.ref_line_smoother import main\nif __name__ == '__main__':\n    sys.exit(main())\n"
)
install(PROGRAMS "${CMAKE_CURRENT_BINARY_DIR}/ref_line_smoother_node"
  DESTINATION lib/${PROJECT_NAME}
)

# planning_base
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/planning_base"
  CONTENT "#!/usr/bin/env python3\nimport sys\nfrom modified_EM_planner.planning_base import main\nif __name__ == '__main__':\n    sys.exit(main())\n"
)
install(PROGRAMS "${CMAKE_CURRENT_BINARY_DIR}/planning_base"
  DESTINATION lib/${PROJECT_NAME}
)

# speed_planner
file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/speed_planner"
  CONTENT "#!/usr/bin/env python3\nimport sys\nfrom modified_EM_planner.speed_planner import main\nif __name__ == '__main__':\n    sys.exit(main())\n"
)
install(PROGRAMS "${CMAKE_CURRENT_BINARY_DIR}/speed_planner"
  DESTINATION lib/${PROJECT_NAME}
)

# file(GENERATE) creates files without the executable bit; --symlink-install then
# points lib/pkg/exe -> build/.../exe, and ros2 run skips non-X_OK paths ("No executable found").
add_custom_target(planning_entrypoints_executable ALL
  COMMAND chmod a+x
          "${CMAKE_CURRENT_BINARY_DIR}/global_path_planner_node"
          "${CMAKE_CURRENT_BINARY_DIR}/ref_line_smoother_node"
          "${CMAKE_CURRENT_BINARY_DIR}/planning_base"
          "${CMAKE_CURRENT_BINARY_DIR}/speed_planner"
  COMMENT "chmod +x planning ros2 run entry points")

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
