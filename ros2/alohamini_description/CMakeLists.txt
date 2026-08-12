cmake_minimum_required(VERSION 3.8)
project(alohamini_description)

find_package(ament_cmake REQUIRED)

set(ALOHAMINI_ASSET_SOURCE
  "${CMAKE_CURRENT_SOURCE_DIR}/../../src/lerobot/robots/alohamini/assets/alohamini2pro"
)
set(ALOHAMINI_RUNTIME_ASSET_SOURCE
  "${CMAKE_CURRENT_SOURCE_DIR}/../../alohamini_sim/data_engine/agents/aloha_mini/assets/alohamini2pro"
)
if(NOT EXISTS "${ALOHAMINI_ASSET_SOURCE}/kinematics.json")
  message(FATAL_ERROR "Run scripts/sync_alohamini_kinematics_assets.py before colcon build")
endif()
if(NOT EXISTS "${ALOHAMINI_RUNTIME_ASSET_SOURCE}/alohamini2pro_meshes")
  message(FATAL_ERROR "The synchronized AlohaMini2Pro runtime mesh mirror is missing")
endif()

install(
  DIRECTORY "${ALOHAMINI_ASSET_SOURCE}/"
  DESTINATION "share/${PROJECT_NAME}/alohamini2pro"
)
install(
  DIRECTORY
    "${ALOHAMINI_RUNTIME_ASSET_SOURCE}/alohamini2pro_meshes"
    "${ALOHAMINI_RUNTIME_ASSET_SOURCE}/collision_meshes"
  DESTINATION "share/${PROJECT_NAME}/alohamini2pro"
)
install(
  DIRECTORY launch
  DESTINATION "share/${PROJECT_NAME}"
  PATTERN "__pycache__" EXCLUDE
)

ament_package()
