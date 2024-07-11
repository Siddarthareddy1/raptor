// Copyright (c) 2015-2017, RAPtor Developer Team
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef RAPTOR_HPP
#define RAPTOR_HPP

#include "raptor-sparse.hpp"

// External 
#ifdef USING_HYPRE
    #include "external/hypre_wrapper.hpp"
#endif
#ifdef USING_MFEM
    #include "external/mfem_wrapper.hpp"
#endif
#ifdef USING_PETSC
    #include "external/petsc_wrapper.hpp"
#endif

// RugeStuben classes
#include "ruge_stuben/cf_splitting.hpp"
#include "ruge_stuben/interpolation.hpp"
#include "ruge_stuben/ruge_stuben_solver.hpp"
#ifndef NO_MPI
    #include "ruge_stuben/par_cf_splitting.hpp"
    #include "ruge_stuben/par_interpolation.hpp"
    #include "ruge_stuben/par_ruge_stuben_solver.hpp"
#endif

// SmoothedAgg classes
#include "aggregation/mis.hpp"
#include "aggregation/aggregate.hpp"
#include "aggregation/candidates.hpp"
#include "aggregation/prolongation.hpp"
#include "aggregation/smoothed_aggregation_solver.hpp"
#ifndef NO_MPI
    #include "aggregation/par_mis.hpp"
    #include "aggregation/par_aggregate.hpp"
    #include "aggregation/par_candidates.hpp"
    #include "aggregation/par_prolongation.hpp"
    #include "aggregation/par_smoothed_aggregation_solver.hpp"
#endif

// AMG multilevel classes
#include "multilevel/multilevel.hpp"
#include "multilevel/level.hpp"
#ifndef NO_MPI
    #include "multilevel/par_multilevel.hpp"
    #include "multilevel/par_level.hpp"
#endif 

// Krylov methods
#include "krylov/cg.hpp"
#include "krylov/par_cg.hpp"
#include "krylov/bicgstab.hpp"
#include "krylov/par_bicgstab.hpp"

// Relaxation methods
#include "precondition/relax.hpp"
#ifndef NO_MPI
    #include "precondition/par_relax.hpp"
#endif

// Preconditioning Methods
#ifndef NO_MPI
    #include "precondition/par_diag_scale.hpp"
#endif


#endif

