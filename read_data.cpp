
#include "read_data.hpp" // read_data header file
#include <Eigen/Dense>  // Eigen Library

#include <vector>       // for vector operations
#include <string>       // for string operations
#include <iostream>     // input output operation: cout
#include <fstream>      // file stream operation: ifstream
#include <sstream>      // string stream operation: istringstream
#include <algorithm>    // replace functionality


/* 
Input: dataset file
Output: returns dataset, data in two-dimensional vector.
*/

using namespace std;

//function to read any dataset with all numerical values like MNIST dataset.
void read_data(string inputfile, vector<vector<double> > &X, vector<double> &Y) {

  ifstream fin;                     // declaring the input file stream
  fin.open(inputfile);              // opening the inputfile

  int l = 0;                        // declaring a integer to track the number of line
  string line;                      // declaring a string to hold the read line of the input file

  if (fin.is_open()) {              // if the input file is open
    cout << "File opened successfully " << endl; 

    while (getline(fin, line)){     // storing the line of input file on the variable line
      l++;                          // increasing the line read counter
      istringstream linestream(line); // converting the read line into an string stream
      vector <double> row;           // declaring a vector to store the current row

      int val = 0;                 // declaring a variable to track the number of values in a row
      while (linestream) {         // while the string stream is not null
        string row_value;          // declaring a string to hold the row values

        if (!getline(linestream, row_value, ',')) // storing the values from stream into row_value one by one
          break;                                  // at the end of row break the while loop
        try { 
          if (val < 784) {                                
            row.push_back(stod(row_value));         // pushing the current value into the row for X values
            val++;
          }
          else if (val == 784)                      // pushing the current value into the Y for y values
          {
            Y.push_back(stod(row_value));
          }
        }
        catch (const invalid_argument err) {      // if there is a error catch the error and display it
          cout << "Invalid value found in the file: " << inputfile << " line: " << l << " value: " << val << endl;
          err.what();
        }
      }

      X.push_back(row);                     // pushing the row into the dataset
      row.clear();                                // clearing the row vector to store the next row
    }
    cout << "Lines read successfully: " << l << endl;                            // displaying the number or lines reads from the input file
  }
  else{
    cout << "Unable to open the specified file " << endl; // output if file can't be opened
  }
}