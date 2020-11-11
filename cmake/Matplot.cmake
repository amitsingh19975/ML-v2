find_package(Matplot++ REQUIRED)
# replace with the matplot lib
# find_package(TIFF REQUIRED)
# find_package(PNG REQUIRED)
# find_package(JPEG REQUIRED)

# include_directories(SYSTEM ${TIFF_INCLUDE_DIR} ${PNG_INCLUDE_DIR} ${JPEG_INCLUDE_DIR})
# set(MATPLOT_LIB "/usr/local/lib/libmatplot.a" ${TIFF_LIBRARIES} ${PNG_LIBRARIES} ${JPEG_LIBRARIES})
set(MATPLOT_LIB Matplot++::matplot)