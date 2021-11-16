    cout << X_train.rows() << "," << X_train.cols() << endl;
    cout << X_test.rows() << "," << X_test.cols() << endl;
    cout << Y_train.rows() << "," << Y_train.cols() << endl;
    cout << Y_train_onehot.rows() << "," << Y_train_onehot.cols() << endl;
    cout << Y_test.rows() << "," << Y_test.cols() << endl;
    cout << Y_test_onehot.rows() << "," << Y_test_onehot.cols() << endl;
    cout << w_1.rows() << "," << w_1.cols() << endl;
    cout << w_2.rows() << "," << w_1.cols() << endl;


    cout << Y_train.block(0,0,10,1) << endl;
    cout << Y_train_onehot.block(0,0,10,10) << endl;
    cout << w_1.block(0,0,10,10) << endl;
    cout << w_2.block(0,0,10,10) << endl;

    // cout << X_train.row(2).format(CleanFmt) <<endl;