#include "neural_network.hpp"

#include <iostream>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

//====================
// Plain NN TRAINING:
//====================

void PlainNN(MatrixXd X, MatrixXd Y, MatrixXd Y_onehot, MatrixXd &w_1, MatrixXd &w_2)
{
    for(int e = 0; e < NUM_EPOCHS; e ++)
    { 
        cout<< endl <<  "Epoch Number: "<< e+1 << endl;
        double epoch_loss = 0.0;
        double epoch_count = 0.0;

        for(int i = 0; i < int(N_train/B); i ++) //int(N_train/B)
        {   
            // cout << i+1 << ",";
            // cout<< "  Iteration Number: "<< i+1 << endl;
            
            // forward propagation that gives the final output and drelu values of first layer
            MatrixXd X_B = X.block(B*i, 0, B, X.cols());
            MatrixXd Y_B = Y.block(B*i, 0, B, Y.cols());
            MatrixXd Y_onehot_B = Y_onehot.block(B*i, 0, B, Y_onehot.cols());

            MatrixXd drelu_1(B, d_1);
            MatrixXd Z_1(B, d_1);
            MatrixXd Y_hat = ForwardPass(X_B, w_1, w_2, Z_1, drelu_1);
            // cout << drelu_1.row(0) << endl;
            // cout << Z_1.row(0) << endl;
            // MatrixXd test = Y_hat.row(0);
            // cout << Y_hat.row(0) << endl;
            // cout << test.rowwise().sum() << endl;
            // cout << Y_onehot.row(0) << endl;
            // cout << Y(0,0) << endl;

            // loss and accuracy computation
            epoch_loss += ComputeLoss(Y_onehot_B, Y_hat);
            epoch_count += ComputeCount(Y_B, Y_hat);

            // backward propagation

            // delta_2 and delta_1 computation 

            // cout << Y_hat.row(0) << endl;
            // cout << Y_onehot.row(0) << endl;
            MatrixXd delta_2 = Y_hat - Y_onehot_B; // finding delta_2 = (y_hat - y)

            // double eta = ((double) 1/(B*100));
            // cout << "eta: " << eta << endl;
            // MatrixXd w2_temp = delta_2.transpose() * Z_1;
            // cout << "w2_temp" << endl << w2_temp.col(0) << endl;
            // MatrixXd w2_temp1 = ((double) 1/(B*100)) * delta_2.transpose() * Z_1; 
            // cout << "w2_temp" << endl << w2_temp1.col(0) << endl;
            

            // cout << "Delta 2: " << endl << delta_2.row(0) << endl;
            // cout << delta_2.rows() << "," << delta_2.cols() << endl;
            // cout << " w2 col 0: " << endl << w_2.col(0) << endl;
            // cout << w_2.rows() << "," << w_2.cols() << endl;

            MatrixXd del2_w2 = delta_2 * w_2;   // computig delta_2(y_hat - y) * w_2
            // cout << "Delta 2 * w2: " << endl << del2_w2.row(0) << endl;
            // cout << del2_w2.rows() << "," << del2_w2.cols() << endl;

            // cout << "drelu_1: " << endl << drelu_1.row(0) << endl;
            // cout << drelu_1.rows() << "," << drelu_1.cols() << endl;

            MatrixXd delta_1 = del2_w2.cwiseProduct(drelu_1);   // computing (delta_2 * w_2) o drelu_1 = delta_1
            // cout << "delta_1: " << endl << delta_1.row(0) << endl;  
            // cout << delta_1.rows() << "," << delta_1.cols() << endl;


            // weight update function
            w_2 = w_2 - ((double) 1/(B*100)) * delta_2.transpose() * Z_1; // layer 2 weight update
            w_1 = w_1 - ((double) 1/(B*100)) * delta_1.transpose() * X_B; // layer 1 weight update

            // cout << "w_2,after interartion" << i << endl << w_2.col(0) << endl;

        }
        // cout<<endl;
        cout<< "Loss: "<< epoch_loss/N_train << endl;
        cout<< "Accuracy: "<< (epoch_count/N_train) * 100 << " %" << endl;
    }
}

void TestPlainNN(MatrixXd X, MatrixXd Y, MatrixXd Y_onehot, MatrixXd &w_1, MatrixXd &w_2){

    double test_loss = 0.0;
    double test_count = 0.0;
        for(int i = 0; i < int(N_test/B); i ++) //int(N_test/B)
        {   
            // cout << i+1 << ",";
            // cout<< "  Iteration Number: "<< i+1 << endl;
            
            // forward propagation that gives the final output and drelu values of first layer
            MatrixXd X_B = X.block(B*i, 0, B, X.cols());
            MatrixXd Y_B = Y.block(B*i, 0, B, Y.cols());
            MatrixXd Y_onehot_B = Y_onehot.block(B*i, 0, B, Y_onehot.cols());

            MatrixXd drelu_1(B, d_1);
            MatrixXd Z_1(B, d_1);
            MatrixXd Y_hat = ForwardPass(X_B, w_1, w_2, Z_1, drelu_1);
            // cout << drelu_1.row(0) << endl;
            // cout << Z_1.row(0) << endl;
            // MatrixXd test = Y_hat.row(0);
            // cout << Y_hat.row(0) << endl;
            // cout << test.rowwise().sum() << endl;
            // cout << Y_onehot.row(0) << endl;
            // cout << Y(0,0) << endl;

            // loss and accuracy computation
            test_loss += ComputeLoss(Y_onehot_B, Y_hat);
            test_count += ComputeCount(Y_B, Y_hat);
            // cout << "Test count: " << test_count << endl; 
        }
        
        cout << endl <<  "Test Loss: "<< test_loss/N_test << endl;
        cout<< "Test Accuracy: "<< (test_count/N_test) * 100 << " %" << endl;

}