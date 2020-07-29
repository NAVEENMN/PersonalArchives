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

using namespace cv;
using namespace std;

Mat image_fin;
Mat image_src, image_dst, image_sx, image_sy;
int lowThreshold;
int const max_lowThreshold = 10;
string window_sobel_x = "window_sobel_x";
string window_sobel_y = "window_sobel_y";
Mat sx_out, sy_out;

void display_image(string title, const cv::Mat &img){
    namedWindow(title, WINDOW_AUTOSIZE);
    imshow(title, img);
}

/*
void get_gradient(){
    Mat grad_image;
    grad_image.create(sx_out.rows,sx_out.cols,sx_out.type());
    cout <<"gradient"<<endl;
    // get the data from gx, gy and make a new image to show gradient intesity.
    int sx_rows = sx_out.rows;
    int sx_cols = sx_out.cols;
    cout<< "sx"<<sx_out.rows<<"x"<<sx_out.cols<<endl;
    cout<< "sx"<<sy_out.rows<<"x"<<sy_out.cols<<endl;
    for (int j=0; j<sx_rows; j++){
        uchar* sx_data= sx_out.ptr<uchar>(j);
        uchar* sy_data= sy_out.ptr<uchar>(j);// get data from sy for same pixel
        for (int i=0; i< sx_cols; i++){
            //sx_data[i], sy_data[i] --> x_com + i y_com is a vector
            float x_component = sx_data[i];
            float y_component = sy_data[i];
            //float com = x_component *x_component+ y_component+y_component;
            //float magnitude = sqrt(com);
            //magnitude = magnitude; // normalize
            if ((x_component+y_component) > 10) {
                float theta = cvFastArctan(y_component, x_component);
            // map these values to a sine function
                float mapped = (theta * 255)/90;
                grad_image.at<uchar>(j,i)= mapped;
            } else {
                grad_image.at<uchar>(j,i)= 0.0;
            }
            //cout << "mapped"<< mapped << endl;
        }
    }
    display_image("grad image", grad_image);
}
*/

void call_back(int a, void*)
{
    Mat image_to_show_x, image_to_show_y, image_to_show1, image_to_show;
    // kernel 1 - gradient along x axis ( sobel x )
    cv::Mat kernel1(3,3,CV_32F, cv::Scalar(0));
    kernel1.at<float>(0,0)= -1.0;kernel1.at<float>(0,1)= 0,0;kernel1.at<float>(0,2)= 1.0;
    kernel1.at<float>(1,0)= -(a);kernel1.at<float>(1,1)= 0.0;kernel1.at<float>(1,2)= a;
    kernel1.at<float>(2,0)= -1.0;kernel1.at<float>(2,1)= 0.0;kernel1.at<float>(2,2)= 1.0;
    cv::filter2D(image_src, image_to_show_x, image_src.depth(), kernel1);

    // kernel 2 - gradient along y axis ( sobel y )
    cv::Mat kernel2(3,3,CV_32F, cv::Scalar(0));
    kernel2.at<float>(0,0)= -1.0;kernel2.at<float>(0,1)= -(a);kernel2.at<float>(0,2)= -1.0;
    kernel2.at<float>(1,0)= 0.0;kernel2.at<float>(1,1)= 0.0;kernel2.at<float>(1,2)= 0.0;
    kernel2.at<float>(2,0)= 1.0;kernel2.at<float>(2,1)= a;kernel2.at<float>(2,2)= 1.0;
    cv::filter2D(image_src, image_to_show_y, image_src.depth(), kernel2);
    
    // merge both gradients
    addWeighted( image_to_show_x, 1.0, image_to_show_y, 1.0, 0.0, image_to_show1);
    addWeighted( image_to_show1, 1.0, image_src, 1.0, 0.0, image_to_show);
    
    image_to_show.copyTo(sx_out);
    imshow( "test", image_to_show);
}

void blur(const cv::Mat &image, cv::Mat &result){
    cv::Mat kernel(3,3,CV_32F, cv::Scalar(0));
    // assigns kernel values
    kernel.at<float>(0,0)= 0.0625;kernel.at<float>(0,1)= 0.125;kernel.at<float>(0,2)= 0.0625;
    kernel.at<float>(1,0)= 0.125;kernel.at<float>(1,1)= 0.25;kernel.at<float>(1,2)= 0.125;
    kernel.at<float>(2,0)= 0.0625;kernel.at<float>(2,1)= 0.125;kernel.at<float>(2,2)= 0.0625;
    
    cv::filter2D(image, result, image.depth(), kernel);
}


int main(int argc, const char * argv[]) {
    
    Size size(500,500);//the dst image size,e.g.100x100
    image_src = imread("/Users/naveenmysore/Documents/cv_projects/imgs/lenna.png");
    
    if(!image_src.data){
        cout << "image not read" << std::endl;
        return -1;
    }
    cv::cvtColor(image_src, image_dst, CV_BGR2GRAY);
    image_src.copyTo(image_fin);
    
    display_image("source_image" , image_src);
    namedWindow( "test", CV_WINDOW_AUTOSIZE );
    createTrackbar( "Min Threshold:", "test", &lowThreshold, max_lowThreshold, call_back );
    call_back(0, 0);
    
    waitKey(0);
    return 0;
}
