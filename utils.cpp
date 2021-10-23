#include "utils.hpp"

#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

MatrixXd onehot_Encoding(MatrixXd X, int m)
{
  MatrixXd res = MatrixXd::Zero(X.rows(),m);

  for(int i =0; i < X.rows(); i++)
  {
    int index = X(i,0) * 10;
    res(i,index) = (double) 1;
  }

  return res;
}