find_package(Armadillo REQUIRED)
include_directories(SYSTEM ${ARMADILLO_INCLUDE_DIRS})

include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/vendor/Dataframe/include")
