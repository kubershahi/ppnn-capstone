#ifndef DEFINE_HPP
#define DEFINE_HPP

#define SCALING_FACTOR 8192 // of 13 bits, 2^13

#include <Eigen/Dense>

// Parameters for the Neural Network
extern int N_train;   // Number of Training Samples
extern int N_test;    // Number of Testing Samples
extern int d;         // Number of Features
extern int m;         // Number of Output Classes
extern int B;         // Batch Size
extern int NUM_EPOCHS;// Number of Epochs

typedef Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic> MatrixXi64;
typedef Eigen::Matrix<uint64_t, 1, Eigen::Dynamic> RowVectorXi64;
typedef Eigen::Matrix<uint64_t, Eigen::Dynamic, 1> ColVectorXi64;

#endif