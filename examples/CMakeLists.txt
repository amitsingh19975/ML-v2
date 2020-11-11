set(EXAMPLE_LIST_PATH "${CMAKE_CURRENT_SOURCE_DIR}/example_list.cmake")

if ( EXISTS "${EXAMPLE_LIST_PATH}" )
  include("${EXAMPLE_LIST_PATH}")

  foreach(EXAMPLE_FILES ${EXAMPLE_LIST})
    
    string( REPLACE ".cpp" "" example_name ${EXAMPLE_FILES} )
    add_executable( "${example_name}" "${CMAKE_CURRENT_SOURCE_DIR}/${EXAMPLE_FILES}")
    target_link_libraries( "${example_name}" PRIVATE 
                            project_options project_warnings 
                            CONAN_PKG::docopt.cpp ${OPENMP_LIB} ${ARMADILLO_LIBRARIES}
    )
    
  endforeach(EXAMPLE_FILES ${EXAMPLE_LIST})

endif()