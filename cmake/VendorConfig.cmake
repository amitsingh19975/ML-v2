find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

find_package(Python3 COMPONENTS Development NumPy)
include_directories(SYSTEM ${Python3_INCLUDE_DIRS})
include_directories(SYSTEM ${Python3_NumPy_INCLUDE_DIRS})

include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/vendor/plot")
include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/vendor/Dataframe/include")
