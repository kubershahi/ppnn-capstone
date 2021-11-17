#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Dense>
// #include "read_data.hpp"
#include "define.hpp"
#include "utils.hpp"
// #include "neural_network.hpp"

using namespace std;
using namespace Eigen;

int N_train;    // Number of Training Samples
int N_test;     // Number of Testing Samples
int d;          // Number of Features: number of input values
int d_1;         // Number of neurons in the first layer
int m;          // Number of classes in the output layer
int B;          // Batch Size
int NUM_EPOCHS; // Number of Epochs

int main()
{
    cout << "This works" << endl;
}