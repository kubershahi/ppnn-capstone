#ifndef UTILS_HPP
#define UTILS_HPP

#include "define.hpp"
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

MatrixXd OnehotEncoding(MatrixXd X);
double Sigmoid(double x);
double DRelu(double x);
MatrixXd Relu(MatrixXd S_1, MatrixXd &drelu_1);
MatrixXd Softmax(MatrixXd S_2);
double ComputeLoss(MatrixXd Y, MatrixXd Y_hat);
float ComputeAccuracy(MatrixXd Y, MatrixXd Y_hat);
MatrixXd ForwardPass(MatrixXd X, MatrixXd w_1, MatrixXd w_2, MatrixXd &Z1, MatrixXd &drelu_1);

#endif