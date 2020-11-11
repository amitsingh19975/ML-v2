set(project_name "ML_driver")


add_executable(${project_name} main.cpp)

target_link_libraries(
  ${project_name} PRIVATE project_options project_warnings 
                CONAN_PKG::docopt.cpp ${OPENMP_LIB} ${ARMADILLO_LIBRARIES} ${MATPLOT_LIB}
)