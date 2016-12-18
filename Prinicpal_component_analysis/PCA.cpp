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
    
    std::string folder = "/Users/naveenmysore/Documents/cd_something/codes/Prinicpal_component_analysis/data/" + std::to_string (num)+"/*.jpg";
    cv::String path(folder); //select only jpg
    vector<cv::String> fn;
    vector<cv::Mat> data;// each image is a feature
    cv::glob(path,fn,true);
    Mat src6;
    for (size_t k=0; k<fn.size(); ++k)
    {
        cv::Mat im = cv::imread(fn[k]);
        cv::Mat greyMat;
        cv::cvtColor(im, greyMat, CV_BGR2GRAY);
        Mat res;
        if (greyMat.empty()) continue; //only proceed if sucsessful
        res = greyMat.reshape(0,1);
        data.push_back(res);
        src6.push_back(res);
    }
    cv::Mat mu, covar;
    calcCovarMatrix(src6, covar, mu, CV_COVAR_NORMAL | CV_COVAR_ROWS);
    CvMat* evec  = cvCreateMat(784,784,CV_32FC1);
    CvMat* eval  = cvCreateMat(784,1,CV_32FC1);
    CvMat cov = covar;
    cvZero(evec);
    cvZero(eval);
    cvEigenVV(&cov, evec, eval, DBL_EPSILON, 0, 0);
    cv::Mat EV(evec->rows, evec->cols, CV_64FC1, evec->data.fl);
    cvReleaseMat(&evec);
    cvReleaseMat(&eval);
    cout<<"done";
    return EV;
}

int main(int argc, const char * argv[]) {
    
    cv::String path("/Users/naveenmysore/Documents/cd_something/codes/Prinicpal_component_analysis/data/0/*.jpg"); //select only jpg
    for(int l=9; l<10;l++){
        cout<<l<<std::endl;
        string file_name ="/Users/naveenmysore/Documents/cd_something/codes/Prinicpal_component_analysis/learned"+std::to_string(l)+".txt";
        ofstream outputfile;
        outputfile.open(file_name);
        cv::Mat EV = get_PCA(l);
        outputfile<<l<<std::endl;
        cout<<EV.rows;
        cout<<EV.cols;
        for(int j=0; j<25; j++){
            for(int i=0; i<800; i++) outputfile << EV.at<double>(j,i)<<" ";
            outputfile<<std::endl;
        }
        outputfile.close();
    }
    cout<<"done";
    //display_image("source_image" , datatp);
    
    waitKey(0);
    return 0;
}
