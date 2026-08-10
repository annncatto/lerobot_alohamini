cmake_minimum_required(VERSION 3.8)
project(alohamini_description)

find_package(ament_cmake REQUIRED)

set(ALOHAMINI_ASSET_SOURCE
  "${CMAKE_CURRENT_SOURCE_DIR}/../../src/lerobot/robots/alohamini/assets/alohamini2pro"
)
if(NOT EXISTS "${ALOHAMINI_ASSET_SOURCE}/kinematics.json")
  message(FATAL_ERROR "Run scripts/sync_alohamini_kinematics_assets.py before colcon build")
endif()

install(
  DIRECTORY "${ALOHAMINI_ASSET_SOURCE}/"
  DESTINATION "share/${PROJECT_NAME}/alohamini2pro"
)
install(
  DIRECTORY launch
  DESTINATION "share/${PROJECT_NAME}"
  PATTERN "__pycache__" EXCLUDE
)

ament_package()
