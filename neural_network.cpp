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
        cout<< "Epoch Number: "<< e+1 << endl;
        double epoch_loss = 0.0;

        for(int i = 0; i < 1; i ++) //int(N_train/B)
        { 
            cout<< "  Iteration Number: "<< i+1 << endl;

            // forward propagation that gives the final output and drelu values of first layer
            MatrixXd drelu_1(B, d_1);
            MatrixXd Z_1(B, d_1);
            MatrixXd Y_hat = ForwardPass(X.block(B*i, 0, B, X.cols()), w_1, w_2, Z_1, drelu_1);
            // cout << drelu_1.row(0) << endl;
            // cout << Z_1.row(0) << endl;
            // MatrixXd test = Y_hat.row(0);
            // cout << Y_hat.row(0) << endl;
            // cout << test.rowwise().sum() << endl;
            // cout << Y_onehot.row(0) << endl;
            // cout << Y(0,0) << endl;

            // loss and accuracy computation
            epoch_loss += ComputeLoss(Y_onehot.block(B*i, 0, B, Y_onehot.cols()), Y_hat);
            float acc = ComputeAccuracy(Y.block(B*i,0,B,Y.cols()), Y_hat);
            cout << "  Loss: " << epoch_loss/(B*(i+1)) << endl;
            cout << "  Accuracy: " << acc << "%" << endl;

            // backward propagation

            // delta_2 and delta_1 computation 
            


            // weight update function




            // MatrixXd D = YY - Y.block(B * i,0,B,Y.cols()); // D = X_B_i.w - Y_B_i
            // //cout<< "diff: "<< endl << D << endl;
            // // Loss Computation
            // MatrixXd loss = D.transpose() * D;
            // MatrixXd delta = X.transpose().block(0,B * i,X.cols(),B) * D; // delta = X^T_B_i(X.w - Y)
            // //cout<< "grad: " << endl << delta << endl;
            // w = w - (delta / (B*100)); // w -= alpha/B * delta
            // //cout<<"weights: "<< endl << w <<endl;
            // //cout<<w<<endl;
            // epoch_loss += loss(0,0);
        }
        // cout<<endl;
        // cout<< "Loss: "<< epoch_loss/N << endl;
    }
}