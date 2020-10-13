find_package(Boost 1.71.0)

if(BOOST_FOUND)
    include_directories(SYSTEM ${Boost_INCLUDE_DIRS})
    message( STATUS "Boost Found" )
else()
    message( FATAL_ERROR "Please Install Boost" )
endif(BOOST_FOUND)
