#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Dense>


using namespace std;
using namespace Eigen;

double DReLU(double x)
{
    if (x > 0.0){
        return 1;
    }
    else{
        return 0;
    }
}

MatrixXd ReLU(MatrixXd X)
{
    MatrixXd X_DReLU = X.unaryExpr(&DReLU);
    MatrixXd res = X.cwiseProduct(X_DReLU);

    // MatrixXd res = X_DReLU.array() * X.array();
    // res = res.array().abs();
    return res.cwiseAbs();
}

MatrixXd Softmax(MatrixXd X)
{
    MatrixXd res(X.rows(), X.cols());
    MatrixXd X_e = X.array().exp();
    cout << X_e << endl;
    VectorXd X_sum= X_e.rowwise().sum();
    cout << X_sum << endl;

    for (int i = 0; i < X.rows(); i++)
    {
        res.row(i) = X_e.row(i)/X_sum(i,0);
    }

    return res;
}

int main()
{
    cout << "Working " << endl;
    MatrixXd X = MatrixXd::Random(3,3);
    cout << X << endl;

    MatrixXd Y = MatrixXd::Random(3,3);
    cout << Y << endl;

    MatrixXd Z = X * Y;
    cout << Z << endl;

    // VectorXi argmax(X.rows());
    // cout << argmax.rows() << "," << argmax.cols() << endl;
    // for (int i = 0; i < X.rows(); i++){
    //     // cout << i << endl;
    //     X.row(i).maxCoeff(&argmax[i]);
    // }
    // cout << argmax << endl;

    // MatrixXd X_log = X.array().log();
    // cout << X_log << endl;
    // MatrixXd X_mult = X.cwiseProduct(X_log);
    // cout << X_mult.sum() << endl;

    // MatrixXd X_R = ReLU(X);
    // cout << X_R << endl;

    // MatrixXd X_S = Softmax(X);
    // cout << X_S<< endl;

    return 0;
}
