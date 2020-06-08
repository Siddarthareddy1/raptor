#ifndef RAPTOR_KRYLOV_CG_HPP
#define RAPTOR_KRYLOV_CG_HPP

#include "core/types.hpp"
#include "core/matrix.hpp"
#ifndef NO_CUDA
    #include "core/cuda/vector_cuda.hpp"
#else
    #include "core/serial/vector.hpp"
#endif
#include <vector>

using namespace raptor;

void CG(CSRMatrix* A, Vector& x, Vector& b, aligned_vector<double>& res, double tol = 1e-05, int max_iter = -1);

#endif
