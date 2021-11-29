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
double ComputeCount(MatrixXd Y, MatrixXd Y_hat);
MatrixXd ForwardPass(MatrixXd X, MatrixXd w_1, MatrixXd w_2, MatrixXd &Z1, MatrixXd &drelu_1);

uint64_t FloatToUint64(double x);
MatrixXi64 FloatToUint64(MatrixXd X);
double Uint64ToFloat(uint64_t x);
MatrixXd Uint64ToFloat(MatrixXi64 X);

void Share(uint64_t X, uint64_t shares[]);
void Share(MatrixXi64 X, MatrixXi64 shares[]);

uint64_t Rec(uint64_t X, uint64_t Y);
MatrixXi64 Rec(MatrixXi64 X, MatrixXi64 Y);

uint64_t Truncate(uint64_t x, int factor);
MatrixXi64 Truncate(MatrixXi64 X, int factor);


#endif