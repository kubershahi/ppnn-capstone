#ifndef DEFINE_HPP
#define DEFINE_HPP

/* 

2^13 = 8192
2^16 = 65536
2^20 = 1048576
2^24 = 16777216

*/

#define SCALING_FACTOR 1048576
#define PRIME_NUMBER 67

#include <Eigen/Dense>

// Parameters for the Neural Network
extern int N_train;   // Number of Training Samples
extern int N_test;    // Number of Testing Samples
extern int d;         // Number of Features
extern int d_1;       // Number of neurons in the first layer
extern int m;         // Number of Output Classes in the last layer
extern int B;         // Batch Size
extern int NUM_EPOCHS;// Number of Epochs

typedef uint64_t myType;
typedef uint8_t smallType;

typedef Eigen::Matrix<uint64_t, Eigen::Dynamic, Eigen::Dynamic> MatrixXi64;
typedef Eigen::Matrix<uint64_t, 1, Eigen::Dynamic> RowVectorXi64;
typedef Eigen::Matrix<uint64_t, Eigen::Dynamic, 1> ColVectorXi64;

#endif