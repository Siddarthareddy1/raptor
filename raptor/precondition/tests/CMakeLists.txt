add_executable(test_jacobi_aniso test_jacobi_aniso.cpp)
target_link_libraries(test_jacobi_aniso raptor ${MPI_LIBRARIES} googletest pthread )
add_test(AnisoJacobiTest ./test_jacobi_aniso)

add_executable(test_jacobi_laplacian test_jacobi_laplacian.cpp)
target_link_libraries(test_jacobi_laplacian raptor ${MPI_LIBRARIES} googletest pthread )
add_test(LaplaceJacobiTest ./test_jacobi_laplacian)

add_executable(test_gs_aniso test_gs_aniso.cpp)
target_link_libraries(test_gs_aniso raptor ${MPI_LIBRARIES} googletest pthread )
add_test(AnisoGSTest ./test_gs_aniso)

add_executable(test_gs_laplacian test_gs_laplacian.cpp)
target_link_libraries(test_gs_laplacian raptor ${MPI_LIBRARIES} googletest pthread )
add_test(LaplaceGSTest ./test_gs_laplacian)

# CANNOT CURRENTLY RUN THESE TESTS, BUT RAPTOR SEEMS CORRECT
# TODO : UNCOMMENT WHEN PYAMG BUG IS FIXED
#
#add_executable(test_sor_aniso test_sor_aniso.cpp)
#target_link_libraries(test_sor_aniso raptor ${MPI_LIBRARIES} googletest pthread )
#add_test(AnisoSORTest ./test_sor_aniso)
#
#add_executable(test_sor_laplacian test_sor_laplacian.cpp)
#target_link_libraries(test_sor_laplacian raptor ${MPI_LIBRARIES} googletest pthread )
#add_test(LaplaceSORTest ./test_sor_laplacian)

#add_executable(test_bsr_jacobi_aniso test_bsr_jacobi_aniso.cpp)
#target_link_libraries(test_bsr_jacobi_aniso raptor ${MPI_LIBRARIES} googletest pthread )
#add_test(BSRAnisoJacobiTest ./test_bsr_jacobi_aniso)

add_executable(test_bsr_jacobi_aniso test_bsr_jacobi_aniso.cpp)
target_link_libraries(test_bsr_jacobi_aniso raptor ${MPI_LIBRARIES} googletest pthread )
add_test(BSRAnisoJacobiTest ./test_bsr_jacobi_aniso)

add_executable(test_bsr_jacobi_laplacian test_bsr_jacobi_laplacian.cpp)
target_link_libraries(test_bsr_jacobi_laplacian raptor ${MPI_LIBRARIES} googletest pthread )
add_test(BSRLaplaceJacobiTest ./test_bsr_jacobi_laplacian)

add_executable(test_bsr_gs_aniso test_bsr_gs_aniso.cpp)
target_link_libraries(test_bsr_gs_aniso raptor ${MPI_LIBRARIES} googletest pthread )
add_test(BSRAnisoGSTest ./test_bsr_gs_aniso)

add_executable(test_bsr_gs_laplacian test_bsr_gs_laplacian.cpp)
target_link_libraries(test_bsr_gs_laplacian raptor ${MPI_LIBRARIES} googletest pthread )
add_test(BSRLaplaceGSTest ./test_bsr_gs_laplacian)

if (WITH_MPI)
    #TODO Add Parallel Relaxation Tests Here

    # TODO Add Parallel Diagonal Scale Tests Here 
endif()

