#include <iostream>
#include <string>
#include <vector>
#include <bitset>
#include <numeric>


#include <Eigen/Dense>

#define BIT_SIZE 64

typedef uint8_t smallType;
typedef uint64_t myType;



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
    // MatrixXd X = MatrixXd::Random(3,3);
    // cout << X << endl;

    // MatrixXd Y = MatrixXd::Random(3,3);
    // cout << Y << endl;

    // MatrixXd Z = X * Y;
    // cout << Z << endl;

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


    // decimal to binary conversion

    // string binary = bitset<64>(128).to_string(); //to binary
    // cout << binary << endl;
    // cout << binary[54] << endl;
    // cout << binary[56] << endl;

    // unsigned long decimal = bitset<64>(binary).to_ulong();
    // cout << decimal<<"\n";

    // cout << sizeof(myType)*CHAR_BIT << endl; 

    // for (int k = 0; k < BIT_SIZE; k++ )
    // {
    //     cout << ((a >> (BIT_SIZE - 1 - k)) & 1) ;
    // }
    // cout << endl;


    // uint64_t a = -45;
    // cout << "a: " << a << endl;
    // uint64_t b = 234;
    // cout << "b: " << b <<endl;
    // uint64_t c = a^b;
    // cout << endl;


    // string a_s = bitset<64>(a).to_string();     // getting the binary representation string
    // cout << "a: " << a_s << endl;

    // vector<int> a_bits;                     
    // for (int i = 0; i < a_s.length(); i ++ )    // converting the binary string to int array
    // {
    //     a_bits.push_back((int(a_s[i]) - 48));
    // }

    // cout << "a: ";
    // for (int i = 0; i < a_bits.size(); i ++ )
    // {
    //     cout << a_bits[i];
    // }
    // cout << endl;


    // string b_s = bitset<64>(b).to_string();
    // cout << "b: " << b_s << endl;

    // vector<int> b_bits;                     
    // for (int i = 0; i < b_s.length(); i ++ )    // converting the binary string to int array
    // {
    //     b_bits.push_back((int(b_s[i]) - 48));
    // }


    // string c_s = bitset<64>(c).to_string();
    // cout << "c: " << c_s << endl;

    // vector<int> c_bits;                     
    // for (int i = 0; i < c_s.length(); i ++ )    // converting the binary string to int array
    // {
    //     c_bits.push_back((int(c_s[i]) - 48));
    // }

    // vector<int> res;
    // int H = 0;

    // for (int i = 0; i < 64; i++)
    // {

    //     int temp = (b_bits[i] - a_bits[i]) + 1 + H;
    //     res.push_back(temp);
    //     H = H + c_bits[i];
    
    // }

    // int r = 0;

    // cout << "r: ";
    // for (int i = 0; i < res.size(); i ++ )
    // {
    //     cout << res[i];
    //     if (res[i]==0)
    //     {
    //         r = 1;
    //     }
    // }
    // cout << endl;

    // if (r==0){
    //     cout << endl << "a>b: False" << endl;
    // }
    // else if (r==1){
    //     cout << endl << "a>b: True" << endl;
    // }

    uint64_t x = 10.55
    


    vector<uint64_t> x_0;
    vector<uint64_t> x_1;
    for (int i = 0; i < 64; i++){
        uint64_t temp = rand() % 67;
        uint64_t temp1 = 
        x_0.push_back(temp);
        
    }
    
    cout << r << endl;

    return 0;
}
