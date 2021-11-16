#include "utils.hpp"

#include <iostream>
#include <string>
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

    VectorXd S2_sum= S2_e.rowwise().sum(); // finding the row wise sum 
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

float ComputeAccuracy(MatrixXd Y, MatrixXd Y_hat)
{
    // cout << Y_hat.row(0) << endl;
    // cout << Y_hat.rows() << "," << Y_hat.cols() << endl;

    VectorXi Y_hat_maxVal_idx(Y.rows());    // creating a column vector of dimension (Y.rows x 1) to store indexes

    for(int i = 0; i < Y_hat.rows(); i++)   // for each row in Y_hat
    {
        Y_hat.row(i).maxCoeff(&Y_hat_maxVal_idx[i]);    // storing the index of maximum value for row i at Y_hat_maxVal_idx[i]
    }

    // cout << Y_hat_maxVal_idx << endl;
    // cout << Y << endl;
    // cout << Y_hat_maxVal_idx.rows() << "," << Y_hat_maxVal_idx.cols() << endl;
    // cout << Y.rows() << "," << Y.cols() << endl;

    int count = 0;
    for (int j; j < Y.rows(); j++)
    {
        if (Y_hat_maxVal_idx(j,0)==Y(j,0))
        {
            count += 1;
        }
    }
    // cout << "  Count: " << count << endl;

    float accuracy = ((float) count/Y.rows() * 100.0);
    string res = to_string(accuracy) + " %";

    // cout << "  Accuracy: " << res << endl;


    return accuracy;
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