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
#include <iomanip>

using namespace cv;
using namespace std;

#define learned_eigen_vectors "/Users/naveenmysore/Documents/cd_something/codes/Prinicpal_component_analysis/EV.xml"
#define vectors_in_transformed_space "/Users/naveenmysore/Documents/cd_something/codes/Prinicpal_component_analysis/new_space.xml"
#define test_image "/Users/naveenmysore/Documents/cd_something/codes/Prinicpal_component_analysis/data/8/4.jpg"

Mat image_src;

void display_image(string title, const cv::Mat &img){
    namedWindow(title, WINDOW_AUTOSIZE);
    imshow(title, img);
}

void desp_mat(string name, Mat M){
    cout<<name<<":"<<"size:"<<M.rows<<"X"<<M.cols<<" type:"<<M.type()<<endl;
}

void write_to_file(string file_name, string name, Mat data){
    FileStorage fs;
    fs.open(file_name, FileStorage::WRITE);
    fs << name << data;
    fs.release();
    
}

 Mat ReadMatFromTxt(string filename, string name)
 {
     FileStorage fs2;
     Mat data;
     fs2.open(filename, FileStorage::READ);
     if (fs2.isOpened())
     {
         fs2[name] >> data ;
         fs2.release();
     } else {
         cout<<"File is not opened\n";
     }
     return data;
}

Mat get_PCA(){
     //select only jpg
    vector<cv::String> fn;
    Mat src6;
    for(int cl=0; cl<10; cl++){
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

void MyLine( Mat img, Point start, Point end )
{
    int thickness = 2;
    int lineType = 8;
    line( img,
         start,
         end,
         Scalar( 0, 0, 0 ),
         thickness,
         lineType );
}


int main(int argc, const char * argv[]) {
    
    VideoCapture cap;
    Mat frame;
    if(!cap.open(0))
        return 0;
    int t = 0;

    while(t<100)
    {
        cap >> frame;
        if( frame.empty() ) break; // end of video stream
        //cv::resize(frame, frame, sizeYouWant);
        /// 2.c. Create a few lines
        MyLine( frame, cvPoint(200,200), cvPoint(700,200) );
        MyLine( frame, cvPoint(700,200), cvPoint(700,700) );
        MyLine( frame, cvPoint(700,700), cvPoint(200,700) );
        MyLine( frame, cvPoint(200,700), cvPoint(200,200) );
        
        imshow("this is you, smile! :)", frame);
        
        if( waitKey(1) == 27 ) break;
        t ++;
    }
    
    cv::Rect roi = cv::Rect(200,200,500,500);
    cv::Mat roiImg;
    roiImg = frame(roi);
    roiImg.copyTo(frame);
    
    
    desp_mat("orignal", frame);
    // to gray -> blur -> sobel -> resize
    cv::cvtColor(frame, frame, CV_BGR2GRAY);
    frame =  cv::Scalar::all(255) - frame;
    cv::threshold(frame, frame, 100, 255, cv::THRESH_BINARY);
    Mat m1 = Mat(28,28, CV_64F, cvScalar(0.));
    cv::resize(frame, frame, m1.size());
    //GaussianBlur( frame, frame, Size( 7, 7), 0, 0 );
    //Sobel(frame, frame, CV_32F, 0, 1);
    
    bool learn_new = false; // if false read principal components from file
    cv::Mat EV;
    cv::Mat new_space;
    if (learn_new){
        EV = get_PCA();
        write_to_file(learned_eigen_vectors, "EV", EV);
    } else {
        EV = ReadMatFromTxt(learned_eigen_vectors,"EV");
    }
    string file_name = test_image;
    //cv::Mat im = cv::imread(file_name, CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat greyMat;
    cv::Mat im;
    im = frame;
    desp_mat("frame", frame);
    cv::Mat res;
    res = im.reshape(0,1);
    res.convertTo(res, CV_32FC1);
    desp_mat("EV", EV);
    EV.convertTo(EV, CV_32FC1);
    res = res.t();

    // pick top 25 Principal components
    cv::Mat PCS;
    for(int i=0; i<25; i++)
    {
        PCS.push_back(EV.row(i));
    }
    PCS.convertTo(PCS, CV_32FC1);
    
    if (learn_new){
        image_src.convertTo(image_src, CV_32FC1);
        new_space = PCS * image_src; // transform vectors to new space.
        cout<<endl;
        write_to_file(vectors_in_transformed_space, "new_space", new_space);
    } else {
        new_space = ReadMatFromTxt(vectors_in_transformed_space,"new_space");
        new_space.convertTo(new_space, CV_32FC1);
        
    }
    
    Mat test_vector = PCS * res;
    
    desp_mat("pcs", PCS);
    desp_mat("new_space", new_space);// new_space has all the vectors to compare againts
    desp_mat("test_vector", test_vector);
    
    // tranform original vectors to 25 dimensional space
    // (# of images) x 784   784 x 25(EV) = (# of images) x 25
    // tranform test vector to 25 dimensional space
    // 1 x 784   784 x 25(EV) = 1 x 25
    // find distance double dist = norm(a,b,NORM_L2);
    // apply KNN
    Mat vec_a = test_vector.col(0).t();
    int dis = 5000;
    int idx = 0;
    
    Mat ivs;
    for(int i=0; i<new_space.cols; i++){
        Mat vec_b = new_space.col(i).t();
        double dist = norm(vec_a,vec_b,NORM_L2);
        if(dist < dis){
            dis = dist;
            idx = i;
        }
        ivs.push_back(dist);
    }
    cv::Mat dst;
    cv::sortIdx(ivs, dst, CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING);
    
    desp_mat("idx", dst);
    int votes[10]={0};
    for(int i=0; i<dst.rows; i++){
        int idx = dst.at<int>(0,0)/800;
        int current_vote = votes[idx];
        current_vote = current_vote + 1;
        votes[idx] = current_vote;
    }
    
    int cl = 0;
    cl = *std::max_element(votes,votes+10);

    file_name ="/Users/naveenmysore/Documents/cd_something/codes/Prinicpal_component_analysis/classes/p"+std::to_string(cl)+".jpg";
    cv::Mat imd = cv::imread(file_name, CV_LOAD_IMAGE_GRAYSCALE);

    display_image("Prediction" , imd);
    display_image("Read" , frame);

    waitKey(0);
    return 0;
}
