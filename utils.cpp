#include "utils.hpp"

#include <iostream>
#include <string>
#include <cmath>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

// function that changes the target labels into one-hot encoding format
MatrixXd OnehotEncoding(MatrixXd X)
{
    MatrixXd res = MatrixXd::Zero(X.rows(),m);  // creating a matrix of zeroes of X.rows x m(output class) 

    for(int i = 0; i < X.rows(); i++)           // for each row 
    {
        int index = X(i,0);         // the value of the ouput label becomes the index
        res(i,index) = (double) 1;  // where we set it as 1 in its corresponding row in res
    }

    return res;
}

// function that computes sigmoid activation
double Sigmoid(double x)
{
    return (1/(1 + exp(-x)));
}

// function tht computes the derivative of ReLU
double DRelu(double x)
{
    if (x > 0.0){
        return 1;
    }
    else{
        return 0;
    }
}

// function that computes the ReLU activation
MatrixXd Relu(MatrixXd S_1, MatrixXd &drelu_1)
{
    drelu_1 = S_1.unaryExpr(&DRelu);    // computing derivative of relu for each value in matrix
    MatrixXd res = S_1.cwiseProduct(drelu_1);   // element-wise corresponding multiplication

    return res.cwiseAbs();  // returning the absolute value          
}

MatrixXd Softmax(MatrixXd S2)
{
    MatrixXd res(S2.rows(), S2.cols()); // declaring a matrix of S2 dimension to hold the result

    MatrixXd S2_e = S2.array().exp();  // applying e^x to every element in matrix
    // cout << S2_e.row(0) << endl;

    VectorXd S2_sum= S2_e.rowwise().sum(); // finding the row wise sum VectorXd is column vector
    // cout << S2_sum(0,0) << endl;

    for (int i = 0; i < S2.rows(); i++) // for each row in matrix 
    {
        res.row(i) = S2_e.row(i)/S2_sum(i,0); // divide it by its sum
    }

    return res; 
}

// function that computes the error of a iterations using cross entropy loss function
double ComputeLoss(MatrixXd Y, MatrixXd Y_hat)
{   
    // cout << Y_hat.row(0) << endl;
    MatrixXd Y_hat_log = Y_hat.array().log();   // applying log-base e to each element in y_hat
    // cout << Y_hat_log.row(0) << endl;
    // cout << Y.row(0) << endl;

    MatrixXd Y_Y_hat_mult = Y.cwiseProduct(Y_hat_log); // element-wise multilication between y and logy_hat
    // cout << Y_Y_hat_mult.row(0) << endl;
    // cout << Y_Y_hat_mult.sum() << endl;

    return -(Y_Y_hat_mult.sum());   // returning the sum of final matrix 

}

double ComputeCount(MatrixXd Y, MatrixXd Y_hat)
{
    // cout << Y_hat.row(0) << endl;
    // cout << Y_hat.rows() << "," << Y_hat.cols() << endl;

    VectorXd Y_hat_maxVal_idx(Y.rows());    // creating a column vector of dimension (Y.rows x 1) to store indexes

    for(int i = 0; i < Y_hat.rows(); i++)   // for each row in Y_hat
    {
        Y_hat.row(i).maxCoeff(&Y_hat_maxVal_idx[i]);    // storing the index of maximum value for row i at Y_hat_maxVal_idx[i]
    }

    // cout << Y_hat_maxVal_idx(0,0) << endl;
    // cout << Y(0,0) << endl;
    // cout << Y_hat_maxVal_idx.rows() << "," << Y_hat_maxVal_idx.cols() << endl;
    // cout << Y.rows() << "," << Y.cols() << endl;

    double count = 0.0;
    for (int j = 0; j < Y.rows(); j++)
    {
        // cout << Y_hat_maxVal_idx(j,0) << "," << Y(j,0) << endl;
        if (Y_hat_maxVal_idx(j,0)==Y(j,0))
        {
            count += 1;
        }
    }
    // cout << "  Count: " << count << endl;

    return count;
}

MatrixXd ForwardPass(MatrixXd X, MatrixXd w_1, MatrixXd w_2, MatrixXd &Z_1, MatrixXd &drelu_1)
{   
    // cout << X.rows() << "," << X.cols() << endl;
    // cout << w_1.rows() << "," << w_1.cols() << endl;

    MatrixXd S_1 = X * w_1.transpose(); // X(B, d) * w_1(d_1,d), weighted sum of 1st layer
    // cout << S_1.row(0) << endl;

    Z_1 = Relu(S_1, drelu_1);  // applying relu on weight sum S_1 to get the ouput of first layer
    // cout << Z_1.row(0) << endl;

    // cout << Z_1.rows() << "," << Z_1.cols() << endl;
    // cout << w_2.rows() << "," << w_2.cols() << endl;

    MatrixXd S_2 = Z_1 * w_2.transpose(); // Z_1(B,d_1) * w_2(m, d_1), weighted sum of 2nd layer
    // cout << S_2.row(0) << endl;
        
    // cout << S_2.rows() << "," << S_2.cols() << endl;

    MatrixXd Z_2 = Softmax(S_2);    // applying softmax to get final output
    // cout << Z_2.rows() << "," << Z_2.cols() << endl;
    // cout << Z_2.row(0) << endl;

    return Z_2; // returning the final output
}

// function that converts a single number from double to unit64
uint64_t FloatToUint64(double x)
{
    uint64_t res;
    if ( x >= 0)
    {
        res = (uint64_t) (x * SCALING_FACTOR);
    }
    else
    {
        x = abs(x * SCALING_FACTOR);
        res = (uint64_t) pow(2,64) - x;
    }
    return res;
}

// function that converts double Matrix to unit64 Matrix
MatrixXi64 FloatToUint64(MatrixXd X)
{
    MatrixXi64 res(X.rows(), X.cols());

    for (int i = 0; i < X.rows(); i++)
    {
        for (int j = 0; j < X.cols(); j++)
        {
            double x = X(i,j);
            if ( x >= 0)
            {
                res(i,j) = (uint64_t) (x * SCALING_FACTOR);
            }
            else
            {
                x = abs(x * SCALING_FACTOR);
                res(i,j) = (uint64_t) pow(2,64) - x;
            }
        }
    } 
    return res;
}


//function that converts a single unit64 number to double
double Uint64ToFloat(uint64_t x)
{
    double res;
    if (x & (1UL << 63))
    {
        res = - ((double) pow(2,64) - x)/SCALING_FACTOR;
    }
    else
    {
        res = ((double) x)/SCALING_FACTOR;
    }

    return res;
}


//function that coverts unit64 matrix to double matrix
MatrixXd Uint64ToFloat(MatrixXi64 X)
{
    MatrixXd res(X.rows(), X.cols());
    for (int i = 0; i < X.rows(); i++)
    {
        for (int j = 0; j < X.cols(); j++)
        {
            uint64_t x = X(i,j);
            if (x & (1UL << 63))
            {
                res(i,j) = -((double) pow(2,64) - x)/SCALING_FACTOR;
                //cout<< res(i,j) << " is negative"<<endl;
            }
            else
            {
                res(i,j) = ((double) x)/SCALING_FACTOR;
                //cout<< res(i,j) << " is positive"<<endl;
            }
        }
    } 
    return res;
}


//function that creates shares of an integer
void Share(uint64_t X, uint64_t shares[])
{
	uint64_t X_0 = rand();
	shares[0] = X_0;
	shares[1] = X - X_0;
}

//function that creates shares of integers in a matrix
void Share(MatrixXi64 X, MatrixXi64 shares[])
{
	MatrixXi64 X_0 = MatrixXi64::Random(X.rows(),X.cols());
	shares[0] = X_0;
	shares[1] = X - X_0;
}


// For integer numbers
uint64_t Rec(uint64_t X, uint64_t Y)
{
	return X + Y;
}


// For integer matrices
MatrixXi64 Rec(MatrixXi64 X, MatrixXi64 Y)
{
	return X + Y;
}


uint64_t Truncate(uint64_t x, int factor)
{
    uint64_t res;

    if (x & (1UL << 63))
    {
        res = (uint64_t) pow(2,64) - ( (uint64_t)pow(2,64) - x)/factor;
        //cout<< res(i,j) << " is negative"<<endl;
    }
    else
    {
        res = x/factor;
        //cout<< res(i,j) << " is positive"<<endl;
    }

    return res;
}

// function that truncates integer values in a given matrix
MatrixXi64 Truncate(MatrixXi64 X, int factor)
{

    MatrixXi64 res(X.rows(), X.cols());
    for (int i = 0; i < X.rows(); i++)
    {
        for (int j = 0; j < X.cols(); j++)
        {
        uint64_t x = X(i,j);
        if (x & (1UL << 63))
        {
            res(i,j) = (uint64_t) pow(2,64) - ( (uint64_t)pow(2,64) - x)/factor;
            //cout<< res(i,j) << " is negative"<<endl;
        }
        else
        {
            res(i,j) = x/factor;
            //cout<< res(i,j) << " is positive"<<endl;
        }
            
        }
    } 
    return res;
}

// // For 64-integer inputs
// MatrixXi64 MatMult(int i, MatrixXi64 X_0, MatrixXi64 X_1, MatrixXi64 Y_0, MatrixXi64 Y_1)
// { 

// 	MatrixXi64 product(X_0.rows(),Y_0.cols());
// 	if (i == 1) product = -(E * F) + (A * F) + (E * B) + Z;
// 	else if (i == 0) product = (A * F) + (E * B) + Z;
	
// 	return product;
// }
