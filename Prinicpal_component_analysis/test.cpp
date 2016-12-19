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



void display_image(string title, const cv::Mat &img){
    namedWindow(title, WINDOW_AUTOSIZE);
    imshow(title, img);
}


 Mat ReadMatFromTxt(string filename, int rows,int cols)
 {
     double m;
     Mat out = Mat::zeros(26, 784, CV_64FC1);//Matrix to store values
 
     ifstream fileStream(filename);
     int cnt = 0;//index starts from 0
     while (fileStream >> m)
     {
        int temprow = cnt / cols;
        int tempcol = cnt % cols;
        //cout << m<<std::endl;
        out.at<double>(temprow, tempcol) = m;
        cnt++;
     }
     return out;
 }


double get_distance(string file){
    cv::Mat mu, covar, greyMat, res;
    cv::Mat im = cv::imread(file);
    //im.convertTo(im, CV_64FC1);
    cv::cvtColor(im, greyMat, CV_BGR2GRAY);
    res = greyMat.reshape(0,1);
    Mat double_I;
    res.convertTo(double_I, CV_64FC1);//ensure same data type
    vector<double> vec0;
    double_I.copyTo(vec0);
    
    for(int k=0; k<10; k++){
        string fl = "/Users/naveenmysore/Documents/cd_something/codes/Prinicpal_component_analysis/learned"+std::to_string(k)+".txt";
        Mat EVS = ReadMatFromTxt(fl, 25, 784);
        double b = 0.0;
        for(int i=0; i<=25; i++){
            vector<double> vec1;
            EVS.row(i).copyTo(vec1);
            //double dis = norm(cv::Mat(double_I),cv::Mat(EVS.row(i)),NORM_L2);
            b = b + double_I.dot(EVS.row(i));
            //cout<<EVS.row(i)<<endl;
        }
        b = b / 25.0;
        double ang = acos(b*1000);
        cout<<"P:"<<k<<" "<<ang<<std::endl;
    }
    return 0.0;
}

int main(int argc, const char * argv[]) {
    double dis;
    int test = 1;
    int id = 4;
    string file_name ="/Users/naveenmysore/Documents/cd_something/codes/Prinicpal_component_analysis/data/"+std::to_string(test)+"/"+std::to_string(id)+".jpg";
    dis = get_distance(file_name);
    cout<<dis<<std::endl;
    cout<<"done";
    //display_image("source_image" , datatp);
    waitKey(0);
    return 0;
}
