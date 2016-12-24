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
#define test_image "/Users/naveenmysore/Documents/cd_something/codes/Prinicpal_component_analysis/data/5/4.jpg"

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
    
    
    bool get_image_from_camera = true;
    bool learn_new = false;
    cv::Mat im;
    Mat grad_x, grad_y;

    if(get_image_from_camera){
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
    cv::Mat contour;
    cv:Mat src_copy;
    roiImg = frame(roi);
    roiImg.copyTo(frame);
    
    
    desp_mat("orignal", frame);
    src_copy = frame;
    //cv::threshold(frame, frame, 100, 255, cv::THRESH_BINARY);
    Mat m1 = Mat(28,28, CV_64F, cvScalar(0.));
    //GaussianBlur( frame, frame, Size( 7, 7), 0, 0 );
    //Sobel(frame, frame, CV_32F, 0, 1);
    GaussianBlur( frame, frame, Size(3,3), 0, 0, BORDER_DEFAULT );
    cvtColor( frame, frame, CV_BGR2GRAY );
        
    dilate(frame, frame
           , Mat(), Point(-1, -1), 2, 1, 1);
        
        // Create a structuring element
        int erosion_size = 3;
        Mat element = getStructuringElement(cv::MORPH_ERODE,
                                            cv::Size(2 * erosion_size + 1, 2 * erosion_size + 1),
                                            cv::Point(erosion_size, erosion_size) );
        
        // Apply erosion or dilation on the image
        erode(frame,frame,element);  // dilate(image,dst,element);
        
    cv::Canny(frame,frame,10,350);
        
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        findContours( frame, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
        /// Draw contours
        Mat drawing = Mat::zeros( frame.size(), CV_64FC1 );
        for( int i = 0; i< contours.size(); i++ )
        {
            Scalar color = Scalar( 255, 255, 255 );
            //drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
            drawContours( frame, contours, i, color, CV_FILLED, 8, hierarchy );
        }
    
        
        /// Approximate contours to polygons + get bounding rects and circles
        vector<vector<Point> > contours_poly( contours.size() );
        vector<Rect> boundRect( contours.size() );
        vector<Point2f>center( contours.size() );
        vector<float>radius( contours.size() );
        
        for( int i = 0; i < contours.size(); i++ )
        { approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
            boundRect[i] = boundingRect( Mat(contours_poly[i]) );
            minEnclosingCircle( (Mat)contours_poly[i], center[i], radius[i] );
        }
        
        
        /// Draw polygonal contour + bonding rects + circles
        drawing = Mat::zeros( frame.size(), CV_64FC1 );
        for( int i = 0; i< contours.size(); i++ )
        {
            Scalar color = Scalar( 255, 255, 255 );
            drawContours( drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
            // rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
            // circle( drawing, center[i], (int)radius[i], color, 2, 8, 0 );
        }
        
    /// Gradient X
    //cv::Scharr( frame, grad_x, CV_64F, 1, 0, 1, 0, BORDER_DEFAULT );
    /// Gradient Y
    //cv::Scharr( frame, grad_y, CV_64F, 0, 1, 1, 0, BORDER_DEFAULT );
    //
        
    // now frame has image with contor

        //Apply thresholding
        cv::threshold(src_copy, src_copy, 100, 255, cv::THRESH_BINARY_INV);
        cv::cvtColor(src_copy, src_copy, CV_BGR2GRAY);
        addWeighted( src_copy, 0.3, frame, 0.7, 0, frame );
        cv::threshold(frame, frame, 100, 255, cv::THRESH_BINARY);//final
        display_image("before" , frame);
        cv::resize(frame, frame, m1.size(),0,0, INTER_AREA);
    im = frame;
    } else {
        string file_name = test_image;
        im = cv::imread(file_name, CV_LOAD_IMAGE_GRAYSCALE);

    }
    //GaussianBlur( frame, frame, Size( 7, 7), 0, 0 );
    //Sobel(frame, frame, CV_32F, 0, 1);
    
     // if false read principal components from file
    cv::Mat EV;
    cv::Mat new_space;
    if (learn_new){
        EV = get_PCA();
        write_to_file(learned_eigen_vectors, "EV", EV);
    } else {
        EV = ReadMatFromTxt(learned_eigen_vectors,"EV");
    }
    
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
    cv::sortIdx(ivs, dst, CV_SORT_EVERY_COLUMN | CV_SORT_ASCENDING);
    
    desp_mat("idx", dst);
    int votes[10]={0};
    for(int i=0; i<20; i++){
        int idx = dst.at<int>(i,0)/800;
        int current_vote = votes[idx-1];
        current_vote = current_vote + 1;
        votes[idx-1] = current_vote;
    }
    
    int cl = 0;
    int index = 0;
    for(int i=0; i<=9; i++){
        cout<<i<<":"<<votes[i]<<endl;
        if(votes[i] > cl){
            cl = votes[i];
            index = i;
        }
    }

    string file_name ="/Users/naveenmysore/Documents/cd_something/codes/Prinicpal_component_analysis/classes/p"+std::to_string(index+1)+".jpg";
    cv::Mat imd = cv::imread(file_name, CV_LOAD_IMAGE_GRAYSCALE);
    display_image("Prediction" , imd);
    display_image("Read" , im);

    waitKey(0);
    return 0;
}
