#ifndef RAPTOR_KRYLOV_PAR_CG_HPP
#define RAPTOR_KRYLOV_PAR_CG_HPP

#include "core/types.hpp"
#include "core/par_matrix.hpp"
#include "core/par_vector.hpp"
#include "multilevel/par_multilevel.hpp"
#include <vector>

using namespace raptor;

void CG(ParCSRMatrix* A, ParVector& x, ParVector& b, aligned_vector<double>& res, 
        double tol = 1e-05, int max_iter = -1, double* comp_t = NULL, bool tap = false);

void PCG(ParCSRMatrix* A, ParMultilevel* ml, ParVector& x, ParVector& b, 
        aligned_vector<double>& res, double tol = 1e-05, int max_iter = -1,
        double* precond_t = NULL, double* comm_t = NULL);

#endif
