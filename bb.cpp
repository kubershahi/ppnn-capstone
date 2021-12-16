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
    

    cout << endl << "Select Building Blocks (enter corresponsing digit): " << endl;
    cout << endl;
    cout << "\t [1] Mapping and Reversing Mapping functionality: " << endl;
    cout << "\t [2] Truncation Functionality: " << endl;
    cout << "\t [3] Secret Sharing functionality: " << endl;
    cout << "\t [4] Secure Matrix Multiplication functionality: " << endl;
    cout << "\t [5] Private Compare functionality (Unshared Setting): " << endl;
    cout << "\t [6] Private Compare functionality (Shared Setting): " << endl;

    int selection = 0;
    cout << endl << "Enter selection: ";
    cin >> selection; 

    IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]"); // formatting option while printing Eigen Matrices
    cout << fixed;

    if (selection == 1) // Mapping and Reverse Mapping functionality 
    {
        MatrixXd X(2,2);
        X << 456.00, -0.987654, 5.66897, 0.88768;   // Random 2 x 2 Matrix
        cout << endl << "Matrix X: " << endl;
        cout << endl <<  X << endl;

        MatrixXi64 X_i = FloatToUint64(X);          // converting to integer ring  
        cout << endl << "Matrix X mapped to Z_L: " << endl;
        cout << endl << X_i << endl;

        MatrixXd X_f = Uint64ToFloat(X_i);          // converting back to floats
        cout << endl << "Matrix X mapped back to floats: " << endl;
        cout << endl << X_f << endl << endl;

    }
    else if (selection == 2) // Truncation Fucntionality 
    {    
        MatrixXd X = MatrixXd::Random(2,2);         // Random Matrix X
        cout << endl << "Matrix X: " << endl;       
        cout << endl << X << endl;

        MatrixXd Y = MatrixXd::Random(2,2);         // Random Matrix Y
        cout << endl << "Matrix Y: " << endl;
        cout << endl << Y << endl;

        MatrixXd Z = X * Y;                         // computing Z in floats itself
        cout << endl << "Matrix Z (floating point multiplication): " << endl;
        cout << endl << Z  << endl;

        MatrixXi64 X_i = FloatToUint64(X);          // converting X to integer ring
        // cout << endl << "Matrix X mapped " << endl;
        // cout << endl << X_i  << endl;

        MatrixXi64 Y_i = FloatToUint64(Y);          // converting Y to integer ring
        // cout << endl << "Matrix X mapped " << endl;
        // cout << endl << Y_i  << endl;

        MatrixXi64 Z_i = X_i * Y_i;                 // computing Z in ring
        // cout << endl << "Matrix Z (X_i * Y_i) " << endl;
        // cout << endl << Z_i  << endl;

        MatrixXi64 Z_t = Truncate(Z_i, SCALING_FACTOR); // truncating Z after multiplication 
        cout << endl << "Truncated Matrix Z (X,Y mapped, multiplied and then truncated) " << endl;
        cout << endl << Z_t  << endl;
        
        MatrixXd Z_f = Uint64ToFloat(Z_t);              // converting Z back to floats
        cout << endl << "Matrix Z mapped back to floats" << endl;
        cout << endl << Z_f << endl << endl;

    }
    else if (selection == 3) // Secret Sharing functionality  
    {       
        MatrixXd X = MatrixXd::Random(2,2);             // Random Matrix X        
        cout << endl << "Matrix X: " << endl;
        cout << endl << X << endl;

        MatrixXi64 X_i = FloatToUint64(X);              // converting X to integer ring 

        MatrixXi64 shares[2];                           
        Share(X_i,shares);                              // creating shares of X 

        MatrixXi64 X_i0 = shares[0];                    // zeroth share of X
        cout << endl << "Matrix X_0: " << endl;
        cout << endl << X_i0 << endl;

        MatrixXi64 X_i1 = shares[1];                    // first share of X 
        cout << endl << "Matrix X_1: " << endl;
        cout << endl << X_i1 << endl;

        MatrixXi64 X_r = Rec(X_i0, X_i1);               // reconstructiong the shares 
        MatrixXd X_f = Uint64ToFloat(X_r);              // converting back to floats 
        cout << endl << "Matrix X_f: " << endl;
        cout << endl << X_f << endl;

    }
    else if (selection == 4) // Secure Matrix Multiplication Functionality
    {
        MatrixXd X = MatrixXd::Random(3,3);     // Random Matrix X
        cout << endl << "X: " << endl;
        cout << X << endl;

        MatrixXd Y = MatrixXd::Random(3,3);     // Random Matrix Y
        cout << endl << "Y: " << endl;
        cout << Y << endl;

        MatrixXd Z = X * Y;
        cout << endl << "Z (floating point muliplication): " << endl;
        cout << Z << endl;


        MatrixXi64 X_i = FloatToUint64(X);      // converting matrix X to integer 
        // cout << endl << "Matrix X mapped : " << endl;
        // cout << X_i << endl;

        MatrixXi64 Y_i = FloatToUint64(Y);      // converting matrix Y to integer
        // cout << endl << "Matrix Y mapped : " << endl;
        // cout << Y_i << endl;

        // MatrixXi64 Z_i = X_i * Y_i;
        // MatrixXi64 Z_i_t = Truncate(Z_i, SCALING_FACTOR);
        // cout << endl << "Matrix Z (X * Y and then truncated): " << endl;
        // cout << Z_i_t << endl;


        cout << endl << "==== Creating shares of X and Y ====" << endl;

        // Creating shares of matrix X
        MatrixXi64 sharesX[2];
        Share(X_i, sharesX);

        MatrixXi64 X_i0 = sharesX[0];           // 0th share of X
        // cout << endl << "Matrix X (0th share) : " << endl;
        // cout << X_i0 << endl;
        
        MatrixXi64 X_i1 = sharesX[1];           // 1st share of X
        // cout << endl << "Matrix X (1st share) : " << endl;
        // cout << X_i1 << endl;

        // Creating shares of matrix Y
        MatrixXi64 sharesY[2];
        Share(Y_i, sharesY);

        MatrixXi64 Y_i0 = sharesY[0];           // 0th share of Y
        // cout << endl << "Matrix Y (0th share) : " << endl;
        // cout << Y_i0 << endl;
        
        MatrixXi64 Y_i1 = sharesY[1];           // 1st share of Y
        // cout << endl << "Matrix Y (1st share) : " << endl;
        // cout << Y_i1 << endl;


        cout << endl << "==== Triplet Generation (C = A * B) ====" << endl;

        MatrixXi64 triplet_shares[6];
        TripletGeneration(X.rows(), X.cols(), Y.rows(), Y.cols(), triplet_shares);

        MatrixXi64 A_0, A_1, B_0, B_1, C_0, C_1;
        A_0 = triplet_shares[0];
        A_1 = triplet_shares[1];

        B_0 = triplet_shares[2];
        B_1 = triplet_shares[3];

        C_0 = triplet_shares[4];
        C_1 = triplet_shares[5];

        // cout << endl << "A_0: " <<endl;
        // cout << A_0 << endl; 


        cout << endl << "==== Secure Mat Mult (Z = X * Y)====" << endl;

        // masking shares of X
        MatrixXi64 E_0 = X_i0 - A_0;
        MatrixXi64 E_1 = X_i1 - A_1;
        MatrixXi64 E = Rec(E_0, E_1); // masked X

        // masking shares of Y
        MatrixXi64 F_0 = Y_i0 - B_0;
        MatrixXi64 F_1 = Y_i1 - B_1;
        MatrixXi64 F = Rec(F_0, F_1); // masked Y

        // performing secure matrix multiplication for both party 0 and party 1

        MatrixXi64 Z_smult0 = MatMult(0, X_i0, Y_i0, E, F, C_0);    // computing 0th share of Z = X * Y
        MatrixXi64 Z_smult0_t = Truncate(Z_smult0, SCALING_FACTOR); // truncating Z_0 after mulitplication
        MatrixXi64 Z_smult1 = MatMult(1, X_i1, Y_i1, E, F, C_1);    // computing 1st share of Z = X * Y
        MatrixXi64 Z_smult1_t = Truncate(Z_smult1, SCALING_FACTOR);   // truncating Z_1 after mulitplication

        MatrixXi64 Z_smult_t = Rec(Z_smult0_t, Z_smult1_t);         // reconstructing Z
        // cout << endl << "Z_smult_i : " << endl;
        // cout << Z_smult_t << endl;

        MatrixXd Z_f = Uint64ToFloat(Z_smult_t);
        cout << endl << "Z_f: " << endl;
        cout << Z_f << endl;

    }
    else if (selection == 5) // Private Compare Functionality (Unshared Setting)
    {
        cout << endl << "==== Private Compare Functionality (Unshared setting) ====" << endl;

        uint64_t x, r;

        cout << endl << "Enter a number (x): ";
        cin >> x;

        cout << "Enter another number (r): ";
        cin >> r;

        // uint64_t x_i = FloatToUint64(x);
        // uint64_t r_i = FloatToUint64(r);

        // cout << endl << "== After being mapped to Z_L ring ==" << endl;

        cout << endl << "x: " << x << endl;
        cout << "r: " << r << endl;

        cout << endl << "== Comparison (x > r) ==" << endl;

        // string x_s = bitset<64>(x_i).to_string(); // getting the binary representation string
        // cout << "x    : " << x_s << endl;

        // string r_s = bitset<64>(r_i).to_string(); // getting the binary representation string
        // cout << "r    : " << r_s << endl;

        int res = unsharedPrivateCompare(x, r);

        if (res == 0)
        {
            cout << endl << "x > r: False" << endl;
        }
        else if (res == 1)
        {
            cout << endl << "x > r: True" << endl;
        }


        // // difference comparison (experimentation)
        // cout << endl << "== Comparison by difference (x - r > 2^63) ==" << endl;

        // int res1 = PrivateCompareDiff(x_i, r_i);

        // if (res1==0)
        // {
        //     cout << endl << "x > r: True" << endl;
        // }
        // else if (res1==1)
        // {
        //     cout << endl << "x > r: False" << endl;
        // }
    }
    else if (selection == 6) // Private Compare Functionality (Shared Setting)
    {
        cout << endl << "==== Private Compare Functionality (Shared Setting) ====" << endl;

        uint64_t x, r;

        cout << endl << "Enter a integer number (x): ";
        cin >> x;

        cout << "Enter another integer number (r): ";
        cin >> r;

        // cout << endl << "== After being mapped to Z_L ring ==" << endl;

        // uint64_t x_i = FloatToUint64(x);
        // uint64_t r_i = FloatToUint64(r);

        cout << endl << "x: " << x << endl;
        cout << "r: " << r << endl;

        cout << endl << "== Comparison (x > r) ==" << endl;

        int res = PrivateCompare(x,r);

        if (res==0)
        {
            cout << endl << "x > r: False" << endl;
        }
        else if (res==1)
        {
            cout << endl << "x > r: True" << endl;
        }

    }
    else if (selection==7) // Random Testing code
    {
        uint64_t t1 = (1LL << 63) + 10;
        uint64_t t2 = (uint64_t) pow(2,64) - 19;

        cout << endl << "t1: " << t1 << endl;
        cout << endl << "t2: " << t2 << endl;
        
        uint64_t diff = t1 - t2;
        cout << endl << "diff: " << diff << endl;

        uint64_t b = 1LL << 63;
        if (diff>b ){
            cout << "True" << endl;
        }
        else 
        {
            cout << "False" << endl;
        }
    }
    else
    {
        cout << endl << "Invalid Input" << endl;
    }

    return 0;
}