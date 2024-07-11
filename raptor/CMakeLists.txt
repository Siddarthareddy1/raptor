add_subdirectory(precondition)
add_subdirectory(ruge_stuben)
add_subdirectory(aggregation)
add_subdirectory(multilevel)
add_subdirectory(krylov)
add_subdirectory(profiling)
add_subdirectory(external)

if (WITH_MPI)
    set(par_SOURCES
        par_strength.cpp
        )
else()
    set(par_SOURCES
        )
endif()

message(STATUS ${sparse_SOURCES})

add_library(raptor  
    ${precond_SOURCES} ${precond_HEADERS}
	${par_SOURCES} strength.cpp 
	${ruge_stuben_SOURCES} ${ruge_stuben_HEADERS}
        ${aggregation_SOURCES} ${aggregation_HEADERS}
        ${multilevel_SOURCES} ${multilevel_HEADERS} 
        ${krylov_SOURCES} ${krylov_HEADERS} 
        ${profile_SOURCES} ${profile_HEADERS}
        ${external_SOURCES} ${external_HEADERS})

target_link_libraries(raptor PUBLIC raptor-sparse ${MPI_C_LIBRARIES} ${MFEM_LIBRARIES} ${METIS_LIBRARIES} ${HYPRE_LIBRARIES}
    ${MUELU_LIBRARIES} ${PETSC_LIBRARIES} ${PTSCOTCH_LIBRARIES} ${PARMETIS_LIBRARIES} ${EXTERNAL_LIBS})

target_include_directories(raptor
  PUBLIC
  $<INSTALL_INTERFACE:include>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
)

include(GNUInstallDirs)
install(TARGETS raptor
  EXPORT raptorTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES raptor.hpp DESTINATION "include/raptor")
install(FILES ${ext_gallery_HEADERS} DESTINATION "include/raptor/gallery/external")
install(FILES ${precond_HEADERS} DESTINATION "include/raptor/precondition")
install(FILES ${ruge_stuben_HEADERS} DESTINATION "include/raptor/ruge_stuben")
install(FILES ${aggregation_HEADERS} DESTINATION "include/raptor/aggregation")
install(FILES ${multilevel_HEADERS} DESTINATION "include/raptor/multilevel")
install(FILES ${krylov_HEADERS} DESTINATION "include/raptor/krylov")
install(FILES ${profile_HEADERS} DESTINATION "include/raptor/profiling")
install(FILES ${external_HEADERS} DESTINATION "include/raptor/external")
install(EXPORT raptorTargets
  FILE raptorTargets.cmake
  NAMESPACE "raptor::"
  DESTINATION
  ${CMAKE_INSTALL_LIBDIR}/cmake/raptor)
export(EXPORT raptorTargets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/raptorTargets.cmake
  NAMESPACE "raptor::")
export(PACKAGE raptor)

configure_file(${PROJECT_SOURCE_DIR}/config/raptorConfig.cmake.in
  ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/raptorConfig.cmake @ONLY)

install(
  FILES ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/raptorConfig.cmake
  DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/cmake/raptor)

if(ENABLE_UNIT_TESTS)
    add_subdirectory(tests)
    add_subdirectory(precondition/tests)
    add_subdirectory(ruge_stuben/tests)
    add_subdirectory(aggregation/tests)
    add_subdirectory(multilevel/tests)
    add_subdirectory(krylov/tests)
    add_subdirectory(external/tests)
endif()
