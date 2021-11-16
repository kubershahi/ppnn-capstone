#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include "define.hpp"
#include "read_data.hpp"
#include "utils.hpp"
#include "neural_network.hpp"

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
    cout<<"Select Dataset (enter corresponding digit):"<<endl;
    cout<<"\t [1] MNIST"<<endl;

    int selection = 0;
    cout<<"Enter selection: ";
    cin>>selection;

    IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]"); // formatting option while printing Eigen Matrices

    MatrixXd X_train,Y_train,Y_train_onehot, X_test,Y_test, Y_test_onehot, w_1, w_2;

    if (selection==1)
    {
        ::N_train = 1000;   
        ::N_test = 1000;     
        ::d = 784; 
        ::m = 10;
        ::d_1 = 256;         
        ::B = 100;           
        ::NUM_EPOCHS = 1;

        cout<<"Reading Data:"<<endl;
        vector<vector<double> > X_train_load;   // dim: 60000 x 784, 60000 training samples with 784 features
        vector<double> Y_train_load;            // dim: 60000 x 1  , the true label of each training sample

        ReadData("datasets/mnist/mnist_code.csv", X_train_load, Y_train_load);
        cout << "here" << endl;

        MatrixXd X_train_1(N_train, d); 
        MatrixXd Y_train_1(N_train, 1); 

        for (int i = 0; i < N_train; i++)
        {
            X_train_1.row(i) = Map<RowVectorXd>(&X_train_load[i][0], d)/256.0;
            Y_train_1.row(i) = Map<RowVectorXd>(&Y_train_load[i],1);
        }

        // vector<vector<double> > X_test_load;    // dim: 10000 x 784, 10000 testing samples with 784 features
        // vector<double> Y_test_load;             // dim: 10000 x 1  , the true label of each testing sample

        // ReadData("datasets/mnist/mnist_test.csv", X_test_load, Y_test_load);                  // for MNIST dataset

        // MatrixXd X_test_1(N_test, d); // 1000, 784
        // MatrixXd Y_test_1(N_test, 1); // 1000, 1

        // for (int i = 0; i < N_test; i++)
        // {
        //     X_test_1.row(i) = Map<RowVectorXd>(&X_test_load[i][0], d)/256.0;
        //     Y_test_1.row(i) = Map<RowVectorXd>(&Y_test_load[i],1)/10.0;
        // }
        X_train = X_train_1;
        Y_train = Y_train_1;
        // X_test = X_test_1;
        // Y_test = Y_test_1;

        Y_train_onehot = OnehotEncoding(Y_train_1);
        // Y_test_onehot = OnehotEncoding(Y_test_1);

        w_1 = MatrixXd::Random(d_1,d);
        w_2 = MatrixXd::Random(m,d_1);
    }

    //==========================================
    // Plain NN TRAINING:
    //==========================================

    PlainNN(X_train,Y_train,Y_train_onehot, w_1, w_2);

    return 0;
}