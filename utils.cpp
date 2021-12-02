#include "utils.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <bitset>

#include <Eigen/Dense>

smallType additionModPrime[PRIME_NUMBER][PRIME_NUMBER];
smallType multiplicationModPrime[PRIME_NUMBER][PRIME_NUMBER];

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

//function that computes the Softmax activation function
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

// function that computes the number of correct predicted outputs
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


// function that does forward propagation 
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


// function that converts a single unit64 number to double
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


//function that converts unit64 matrix to double matrix
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


// function that creates shares of integers in a matrix
void Share(MatrixXi64 X, MatrixXi64 shares[])
{
	MatrixXi64 X_0 = MatrixXi64::Random(X.rows(),X.cols())/10000;
	shares[0] = X_0;
	shares[1] = X - X_0;
}


// function that reconstructs the share of an integer
uint64_t Rec(uint64_t X, uint64_t Y)
{
	return X + Y;
}


// function that reconstructs the share of an integer in a matrix
MatrixXi64 Rec(MatrixXi64 X, MatrixXi64 Y)
{
	return X + Y;
}


// function that truncates a number by a given factor
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


// function that generates Beavers Triplet 
void TripletGeneration( int X_row , int X_col, int Y_row , int Y_col, MatrixXi64 triplet_shares[])
{
    MatrixXi64 A = MatrixXi64::Random(X_row, X_col)/10000; // masks X
    // cout << endl << "A : " << endl;
    // cout << A << endl;

    MatrixXi64 B = MatrixXi64::Random(Y_row, Y_col)/10000; // masks Y
    // cout << endl << "B : " << endl;
    // cout << B << endl;

    MatrixXi64 C = A * B;
    // cout << endl << "C : " << endl;
    // cout << C << endl;

    MatrixXi64 shares[2];

    Share(A, shares);
    triplet_shares[0] = shares[0]; 
    triplet_shares[1] = shares[1];

    Share(B, shares);
    triplet_shares[2] = shares[0]; 
    triplet_shares[3] = shares[1];

    Share(C, shares);
    triplet_shares[4] = shares[0]; 
    triplet_shares[5] = shares[1];
}

// function that does the Beaver Triplet based secure multiplication
MatrixXi64 MatMult(int i, MatrixXi64 X_s, MatrixXi64 Y_s, MatrixXi64 E, MatrixXi64 F, MatrixXi64 C_s)
{ 

    MatrixXi64 prod_s(X_s.rows(), Y_s.cols());

    if (i == 0)
    {
        prod_s = X_s * F + E * Y_s + C_s;
    }
    else if ( i== 1)
    {
        prod_s = - E * F + X_s * F + E * Y_s + C_s;
    }
	
	return prod_s;
}

vector<int> GetBinaryVector(uint64_t a)
{
    string a_s = bitset<64>(a).to_string(); // getting the binary representation string
    // cout << "a: " << a_s << endl;

    vector<int> a_bits;
    for (int i = 0; i < a_s.length(); i ++ )    // converting the binary string to int array
    {
        a_bits.push_back((int(a_s[i]) - 48));
    }

    return a_bits;
}

int PrivateCompare(uint64_t a, uint64_t b)
{   
    uint64_t diff = (uint64_t) a - b;
    cout << "diff : " << diff;
    vector<int> diff_bits = GetBinaryVector(diff);
    
    cout << endl << "diff : ";
    for (int i = 0; i < diff_bits.size(); i ++ )
    {
        cout << diff_bits[i];
    }
    
    uint64_t boundary = 1LL << 63;
    vector<int> boundary_bits = GetBinaryVector(boundary);

    cout << endl << "bound: ";
    for (int i = 0; i < boundary_bits.size(); i ++ )
    {
        cout << boundary_bits[i];
    }
    cout << endl;

    uint64_t x = diff^boundary;
    vector<int> xor_bits = GetBinaryVector(x);
    cout << "xor  : ";
    for (int i = 0; i < xor_bits.size(); i ++ )
    {
        cout << xor_bits[i];
    }
    cout << endl;

    vector<int> res;
    int H = 0;

    for (int i = 0; i < 64; i++)
    {
        int temp = (boundary_bits[i] - diff_bits[i]) + 1 + H;
        res.push_back(temp);
        H = H + xor_bits[i];
    }

    int r = 0;

    cout << "res  : ";
    for (int i = 0; i < res.size(); i ++ )
    {
        cout << res[i];
        if (res[i]==0)
        {
            r = 1;
        }
    }
    cout << endl << endl << "res(diff > bound)  : " << r << endl;

    return r;
}

int PrivateCompare1(uint64_t a, uint64_t b)
{   
    vector<int> a_bits = GetBinaryVector(a);
    vector<int> b_bits = GetBinaryVector(b);

    uint64_t x = a^b;
    vector<int> xor_bits = GetBinaryVector(x);
    cout << "xor  : ";
    for (int i = 0; i < xor_bits.size(); i ++ )
    {
        cout << xor_bits[i];
    }
    cout << endl;

    vector<int> res;

    int H = 0;

    for (int i = 0; i < 64; i++)
    {

        int temp = (b_bits[i] - a_bits[i]) + 1 + H;
        res.push_back(temp);
        H = H + xor_bits[i];
    
    }

    int r = 0;

    cout << "res  : ";
    for (int i = 0; i < res.size(); i ++ )
    {
        cout << res[i];
        if (res[i]==0)
        {
            r = 1;
        }
    }

    cout << endl << "res(a > b): " << r << endl;

    return r;
}


