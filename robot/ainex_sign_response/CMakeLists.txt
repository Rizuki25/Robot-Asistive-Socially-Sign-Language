cmake_minimum_required(VERSION 3.0.2)
project(ainex_sign_response)
find_package(catkin REQUIRED)
catkin_package()
catkin_install_python(PROGRAMS scripts/response_node.py scripts/ready_audio_node.py scripts/inspect_robot.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY config launch audio DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
