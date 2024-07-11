# Include the directory itself as a path to include directories
set(CMAKE_INCLUDE_CURRENT_DIR ON)
 
#Create a variable called linalg_SOURCES containing all .cpp files:

if (WITH_MPI)
    set(par_precond_HEADERS
        precondition/par_relax.hpp
        precondition/par_diag_scale.hpp
        )
    set(par_precond_SOURCES
        precondition/par_relax.cpp
        precondition/par_diag_scale.cpp
        )
else ()
    set(par_precond_HEADERS
        ""
        )
    set (par_precond_SOURCES 
        ""
        )
endif()

set(precond_HEADERS
    precondition/relax.hpp
    ${par_precond_HEADERS}
    PARENT_SCOPE
    )
set(precond_SOURCES 
    precondition/relax.cpp
    ${par_precond_SOURCES}
    PARENT_SCOPE
    )




