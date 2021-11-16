#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include "define.hpp"
#include "utils.hpp"

#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

// function that trains plain NN
void PlainNN(MatrixXd X, MatrixXd Y, MatrixXd Y_onehot, MatrixXd &w_1, MatrixXd &w_2);

#endif 