#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include "define.hpp"
#include "read_data.hpp"
#include "utils.hpp"

using namespace std;
using namespace Eigen;

int main()
{
    cout<<"Select Dataset (enter corresponding digit):"<<endl;
    cout<<"\t [1] MNIST"<<endl;

    int selection = 0;
    cout<<"Enter selection: ";
    cin>>selection;

    int N_train;    // Number of Training Samples
    int N_test;     // Number of Testing Samples
    int d;          // Number of Features
    int m;          // Number of classes
    int B;          // Batch Size
    int NUM_EPOCHS; // Number of Epochs

    IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]"); // formatting option while printing Eigen Matrices

    MatrixXd X_train,Y_train,Y_train_onehot, X_test,Y_test, Y_test_onehot;

    if (selection==1)
    {
        N_train = 10000;   
        N_test = 1000;     
        d = 784; 
        m = 10;          
        B = 128;           
        NUM_EPOCHS = 1;

        cout<<"Reading Data:"<<endl;
        vector<vector<double> > X_train_load;   // dim: 60000 x 784, 60000 training samples with 784 features
        vector<double> Y_train_load;            // dim: 60000 x 1  , the true label of each training sample

        read_data("datasets/mnist/mnist_train.csv", X_train_load, Y_train_load);

        MatrixXd X_train_1(N_train, d); 
        MatrixXd Y_train_1(N_train, 1); 

        for (int i = 0; i < N_train; i++)
        {
            X_train_1.row(i) = Map<RowVectorXd>(&X_train_load[i][0], d)/256.0;
            Y_train_1.row(i) = Map<RowVectorXd>(&Y_train_load[i],1)/10.0;
        }

        vector<vector<double> > X_test_load;    // dim: 10000 x 784, 10000 testing samples with 784 features
        vector<double> Y_test_load;             // dim: 10000 x 1  , the true label of each testing sample

        read_data("datasets/mnist/mnist_test.csv", X_test_load, Y_test_load);                  // for MNIST dataset

        MatrixXd X_test_1(N_test, d); // 1000, 784
        MatrixXd Y_test_1(N_test, 1); // 1000, 1

        for (int i = 0; i < N_test; i++)
        {
            X_test_1.row(i) = Map<RowVectorXd>(&X_test_load[i][0], d)/256.0;
            Y_test_1.row(i) = Map<RowVectorXd>(&Y_test_load[i],1)/10.0;
        }
        X_train = X_train_1;
        Y_train = Y_train_1;
        X_test = X_test_1;
        Y_test = Y_test_1;

        Y_train_onehot = onehot_Encoding(Y_train_1,m);
        Y_test_onehot = onehot_Encoding(Y_test_1,m);

    }

    cout << X_train.rows() << "," << X_train.cols() << endl;
    cout << X_test.rows() << "," << X_test.cols() << endl;
    cout << Y_train.rows() << "," << Y_train.cols() << endl;
    cout << Y_train_onehot.rows() << "," << Y_train_onehot.cols() << endl;
    cout << Y_test.rows() << "," << Y_test.cols() << endl;
    cout << Y_test_onehot.rows() << "," << Y_test_onehot.cols() << endl;


    cout << Y_train.block(0,0,10,1) << endl;
    cout << Y_train_onehot.block(0,0,10,10) << endl;

    // cout << X_train.row(2).format(CleanFmt) <<endl;
    
    return 0;
}