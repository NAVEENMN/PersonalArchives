//
//  main.cpp
//  Opencvtest
//
//  Created by Naveen Mysore on 9/10/16.
//  Copyright © 2016 naveen_mysore. All rights reserved.
//

// This is the implementation of Sobel kernels in X and Y axis.

#include <iostream>
#include <string>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <fstream>

using namespace cv;
using namespace std;

Mat image_src;


void display_image(string title, const cv::Mat &img){
    namedWindow(title, WINDOW_AUTOSIZE);
    imshow(title, img);
}

void desp_mat(string name, Mat M){
    cout<<name<<":"<<"size:"<<M.rows<<"X"<<M.cols<<" type:"<<M.type()<<endl;
}
/*
 Mat ReadMatFromTxt(string filename, int rows,int cols)
 {
 double m;
 Mat out = Mat::zeros(32, 32, CV_64FC1);//Matrix to store values
 
 ifstream fileStream(filename);
 int cnt = 0;//index starts from 0
 while (fileStream >> m)
 {
 int temprow = cnt / cols;
 int tempcol = cnt % cols;
 //cout << m<<std::endl;
 out.at<int>(temprow, tempcol) = m;
 cnt++;
 }
 return out;
 }
 */

Mat get_PCA(int num){
     //select only jpg
    vector<cv::String> fn;
    Mat src6;
    for(int cl=0; cl<5; cl++){
        std::string folder = "/Users/naveenmysore/Documents/cd_something/codes/Prinicpal_component_analysis/data/" + std::to_string (cl)+"/*.jpg";
        cv::String path(folder);
        cv::glob(path,fn,true);
        for (size_t k=0; k<fn.size(); ++k)
        {
            cv::Mat im = cv::imread(fn[k], CV_LOAD_IMAGE_GRAYSCALE);
            Mat res;
            if (im.empty()) continue; //only proceed if sucsessful
            res = im.reshape(0,1);
            src6.push_back(res);
        }
    }
    src6 = src6.t();
    image_src = src6;
    cv::Mat mu, covar;
    calcCovarMatrix(src6, covar, mu, CV_COVAR_NORMAL | CV_COVAR_COLS);
    CvMat* evec  = cvCreateMat(784,784,CV_64FC1);
    CvMat* eval  = cvCreateMat(784,1,CV_64FC1);
    covar = covar/(src6.rows -1);
    CvMat cov = covar;
    cvZero(evec);
    cvZero(eval);
    cvEigenVV(&cov, evec, eval, DBL_EPSILON, 0, 0);
    cv::Mat EV(evec->rows, evec->cols, CV_64FC1, evec->data.fl);
    cvReleaseMat(&evec);
    cvReleaseMat(&eval);
    return EV;
}

int main(int argc, const char * argv[]) {
    
    cv::String path("/Users/naveenmysore/Documents/cd_something/codes/Prinicpal_component_analysis/data/0/*.jpg"); //select only jpg
    cv::Mat EV;
    for(int l=0; l<1;l++){
        cout<<l<<std::endl;
        string file_name ="/Users/naveenmysore/Documents/cd_something/codes/Prinicpal_component_analysis/learned"+std::to_string(l)+".txt";
        ofstream outputfile;
        outputfile.open(file_name);
        EV = get_PCA(l);
        outputfile<<l<<std::endl;
        for(int j=0; j<25; j++){
            for(int i=0; i<784; i++) outputfile << EV.at<double>(j,i)<<" ";
            outputfile<<std::endl;
        }
        outputfile.close();
    }
    
    string file_name ="/Users/naveenmysore/Documents/cd_something/codes/Prinicpal_component_analysis/data/4/4.jpg";
    cv::Mat im1 = cv::imread(file_name, CV_LOAD_IMAGE_GRAYSCALE);
    cv:Mat im = im1;
    cv::Mat greyMat, res;
    res = im.reshape(0,1);
    res.convertTo(res, CV_32FC1);
    EV.convertTo(EV, CV_32FC1);
    res = res.t();

    // pick top 25 Principal components
    cv::Mat PCS;
    for(int i=0; i<25; i++)
    {
        PCS.push_back(EV.row(i));
    }
    PCS.convertTo(PCS, CV_32FC1);
    image_src.convertTo(image_src, CV_32FC1);
    Mat new_space = PCS * image_src;
    Mat test_vector = PCS * res;
    
    //desp_mat("new_space", new_space);// new_space has all the vectors to compare againts
    // desp_mat("test_vector", test_vector);
    
    // tranform original vectors to 25 dimensional space
    // (# of images) x 784   784 x 25(EV) = (# of images) x 25
    // tranform test vector to 25 dimensional space
    // 1 x 784   784 x 25(EV) = 1 x 25
    // find distance double dist = norm(a,b,NORM_L2);
    // apply KNN
    Mat vec_a = test_vector.col(0).t();
    int dis = 5000;
    int idx = 0;
    for(int i=0; i<new_space.cols; i++){
        Mat vec_b = new_space.col(i).t();
        double dist = norm(vec_a,vec_b,NORM_L2);
        if(dist < dis){
            dis = dist;
            idx = i;
        }
    }
    if (idx <= 800) {
        cout<<"class: "<<"0"<<endl;
    }
    if (idx > 800 && idx <= 1600) {
        cout<<"class: "<<"1"<<endl;
    }
    if (idx > 1600 && idx <= 2400) {
        cout<<"class: "<<"2"<<endl;
    }
    if (idx > 2400 && idx <= 3200) {
        cout<<"class: "<<"3"<<endl;
    }
    if (idx > 3200) {
        cout<<"class: "<<"4"<<endl;
    }


    cout<<"done";
    display_image("source_image" , im1);
    display_image("dst_image" , im);

    waitKey(0);
    return 0;
}
