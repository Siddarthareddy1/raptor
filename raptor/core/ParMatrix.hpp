// Copyright (c) 2015, Raptor Developer Team, University of Illinois at Urbana-Champaign
// License: Simplified BSD, http://opensource.org/licenses/BSD-2-Clause
#ifndef PARMATRIX_HPP
#define PARMATRIX_HPP

#include <mpi.h>
#include <math.h>
#include <Eigen/Dense>

#include "Matrix.hpp"
#include "ParComm.hpp"

using Eigen::VectorXd;

class ParMatrix
{
public:
    ParMatrix(int _GlobRows, int _GlobCols, Matrix* _diag, Matrix* _offd);
    ParMatrix(int _GlobRows, int _GlobCols, int* ptr, int* idx,
             double* data, int* globalRowStarts)
    {
        int rank, numProcs;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

        this->globalRows = _GlobRows;
        this->globalCols = _GlobCols;

        firstColDiag = globalRowStarts[rank];
        this->localRows = globalRowStarts[rank+1] - firstColDiag;
        int lastColDiag = firstColDiag + localRows - 1;

        std::vector<int> diagI;
        std::vector<int> diagJ;
        std::vector<double> diagData;
        std::vector<int> offdI;
        std::vector<int> offdJ;
        std::vector<double> offdData;
        int diagCtr = 0;
        int offdCtr = 0;

        //Split ParMat into diag and offd matrices
        diagI.push_back(0);
        offdI.push_back(0);
        for (int i = 0; i < localRows; i++)
        {
            int rowStart = ptr[i];
            int rowEnd = ptr[i+1];
            for (int j = rowStart; j < rowEnd; j++)
            {
                int globalCol = idx[j];

                //In offd block
                if (globalCol < firstColDiag || globalCol > lastColDiag)
                {
                    offdCtr++;
                    offdJ.push_back(globalCol);
                    offdData.push_back(data[j]);
                }
                else //in diag block
                {
                    diagCtr++;
                    diagJ.push_back(globalCol - firstColDiag);
                    diagData.push_back(data[j]);
                }
            }
            diagI.push_back(diagCtr);
            offdI.push_back(offdCtr);
        }

        for (int i = 0; i < localRows; i++)
        {
            int row_start = diagI[i];
            int row_end = diagI[i+1];
            for (int j = row_start; j < row_end; j++)
            {
                //printf("Diag(%d, %d) = %2.3e\n", i+firstColDiag, diagJ[j]+firstColDiag, diagData[j]);
            }
        }

        //Create localToGlobal map and convert offd
        // cols to local (currently global)
        int offdNNZ = offdJ.size();
        offdNumCols = 0;
        if (offdNNZ)
        {
            //New vector containing offdCols
            std::vector<int> offdCols;
            for (int i = 0; i < offdI[localRows]; i++)
            {
                offdCols.push_back(offdJ[i]);
            }

            //Sort columns
            std::sort(offdCols.begin(), offdCols.end());

            //Count columns with nonzeros in offd
            offdNumCols = 1;
            for (int i = 0; i < offdNNZ-1; i++)
            {
                if (offdCols[i+1] > offdCols[i])
                {
                    offdCols[offdNumCols++] = offdCols[i+1];
                }
            }

            //Map global columns to indices: 0 - offdNumCols
            for (int i = 0; i < offdNumCols; i++)
            {
                this->localToGlobal.push_back(offdCols[i]);
                this->globalToLocal[offdCols[i]] = i;
            }

            //Convert offd cols to local
            for (int i = 0; i < localRows; i++)
            {
                int rowStart = offdI[i];
                int rowEnd = offdI[i+1];
                for (int j = rowStart; j < rowEnd; j++)
                {
                    offdJ[j] = globalToLocal[offdJ[j]];
                }
            }

            //Initialize two matrices (offd and diag)
            offd = new Matrix(offdI.data(), offdJ.data(), offdData.data(),
                          localRows, offdNumCols, offdData.size());
            (offd->m)->makeCompressed();
        }

        diag = new Matrix(diagI.data(), diagJ.data(), diagData.data(),
                          localRows, localRows, diagData.size());
        (diag->m)->makeCompressed();

        //Initialize communication package
        comm = new ParComm(offd, localToGlobal, globalRowStarts);


    }
    ParMatrix(ParMatrix* A);
    ~ParMatrix();

    int globalRows;
    int globalCols;
    int localRows;
    Matrix* diag;
    Matrix* offd;
    std::vector<int> localToGlobal;
    std::map<int, int> globalToLocal;
    int offdNumCols;
    int firstColDiag;

    ParComm* comm;
};
#endif
