#include "par_air_solver.hpp"

namespace raptor {

template<class T>
void ParAIRSolver::extend_hier(T & A) {
	int level_ctr = levels.size() - 1;
	bool tap_level = tap_amg >= 0 && tap_amg <= level_ctr;
	splitting_t split;
	auto S = A.strength(strength_type, strong_threshold, tap_level,
	                    num_variables, variables);

	split_rs(S, split.on_proc, split.off_proc, tap_level);
	auto P = one_point_interpolation(A, *S, split);
	auto R = local_air(A, *S, split, fpoint_distance::two);

	levels.back()->P = P;
	levels.back()->R = R;
	levels.emplace_back(new ParLevel());
	auto & level = *levels.back();
	auto AP = A.mult(P);
	auto coarse_A = R->mult(AP);

	level.A = coarse_A;

	finalize_coarse_op(tap_amg >= 0 && tap_amg <= level_ctr + 1);
	init_coarse_vectors();

	level.P = nullptr;

	delete AP;
	delete S;
}

void ParAIRSolver::extend_hierarchy()
{
	int level_ctr = levels.size() - 1;
	ParCSRMatrix *A = levels[level_ctr]->A;
	ParBSRMatrix *A_bsr = dynamic_cast<ParBSRMatrix*>(A);
	if (A_bsr) extend_hier(*A_bsr);
	else extend_hier(*A);
}

void ParAIRSolver::finalize_coarse_op(bool tap_init) {
	auto & coarse_op = *levels.back()->A;
	auto & fine_op = *(**(levels.end() - 2)).A;

	coarse_op.sort();
	coarse_op.on_proc->move_diag();
	coarse_op.comm = new ParComm(coarse_op.partition, coarse_op.off_proc_column_map,
	                             coarse_op.on_proc_column_map,
	                             fine_op.comm->key, fine_op.comm->mpi_comm);

	if (tap_init) {
		coarse_op.init_tap_communicators(RAPtor_MPI_COMM_WORLD);
	}
}

void ParAIRSolver::init_coarse_vectors() {
	auto & clevel = *levels.back();
	[global_rows = clevel.A->global_num_rows,
	 local_rows = clevel.A->local_num_rows](auto & ... vecs) {
		(vecs.resize(global_rows, local_rows),...);
	}(clevel.x, clevel.b, clevel.tmp);
}


}
