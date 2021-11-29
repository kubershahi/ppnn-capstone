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
    IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]"); // formatting option while printing Eigen Matrices

    cout << fixed;


    // double X = -10.15723545348;
    // double Y = 5.23423452345;

    // cout << endl << "X: " << X << endl;
    // cout << "Y: " << Y << endl;

    // double Z = X * Y;
    // cout << endl << "Z (X * Y: floating arithmetic): " << Z << endl << endl;

    // uint64_t X_i = FloatToUint64(X); // mapping X to integer
    // uint64_t Y_i = FloatToUint64(Y); // mapping Y to integer

    // // secret sharing setting
    // cout << "=== Secret Sharing Setting (creating shares of X and Y, multiplying and then truncating them, then recreating Z) ==="<<endl << endl;

    // uint64_t shares[2];
    // Share(X_i, shares);               // creating shares of XX
    // uint64_t X_i0 = shares[0];
    // uint64_t X_i1 = shares[1];

    // Share(Y_i, shares);               // creating shares of XX
    // uint64_t Y_i0 = shares[0];
    // uint64_t Y_i1 = shares[1];

    // uint64_t Z_i0 = X_i0 * Y_i0 + X_i0 * Y_i1;  // 0th share of Z
    // uint64_t Z_i1 = X_i1 * Y_i0 + X_i1 * Y_i1;  // 1st share of Z

    // // truncating both the shares, and then recreating
    // uint64_t Z_i0_t = Truncate(Z_i0, SCALING_FACTOR);
    // // cout << "Truncated 0th share of Z: " << Z_i0_t << endl;

    // uint64_t Z_i1_t = Truncate(Z_i1, SCALING_FACTOR);
    // // cout << "Truncated 1st share of Z: " << Z_i1_t << endl;


    // uint64_t Z_sha_r = Rec(Z_i0_t, Z_i1_t);
    // // cout << endl << "Z truncated (shared setting): " << Z_sha_r << endl;

    // double Z_sha_f = Uint64ToFloat(Z_sha_r);
    // cout << "Z (shared setting): " << Z_sha_f << endl;

    cout << endl << " Matrix Test " << endl;

    MatrixXd Test1 = MatrixXd::Random(3,3);
    cout << endl << Test1 << endl;

    MatrixXi64 Test1_i = FloatToUint64(Test1);
    cout << endl << Test1_i << endl;


    MatrixXi64 shares1[2];
    Share(Test1_i, shares1);
    MatrixXi64 Test1_i0 = shares1[0];
    cout << endl << Test1_i0 << endl;
    MatrixXi64 Test1_i1 = shares1[1];
    cout << endl << Test1_i1 << endl;

    MatrixXi64 Test1_rec = Rec(Test1_i0, Test1_i1);
    cout << endl << Test1_rec << endl;



    MatrixXd Test1_f = Uint64ToFloat(Test1_i);
    cout << endl << Test1_f << endl;

    // MatrixXd Test2 = MatrixXd::Random(3,3);
    // cout << endl << Test2 << endl;

    

    // MatrixXi64 Test2_uint = FloatToUint64(Test2);
    // cout << endl << Test2_uint << endl;

    


    // MatrixXd Test2_f = Uint64ToFloat(Test2_uint);
    // cout << endl << Test2_f << endl;

    

    return 0;
}



// cout << "========= Single Test ========" << endl;
    // MatrixXd t(1,1);
    // t << 0.511210;
    // cout << t << endl;

    // MatrixXi64 t_uint = FloatToUint64(t);
    // cout << t_uint << endl;

    // MatrixXd t_int = Uint64ToFloat(t_uint);
    // cout << t_int << endl;