//
//  main.cpp
//  Opencvtest
//
//  Created by Naveen Mysore on 9/10/16.
//  Copyright © 2016 naveen_mysore. All rights reserved.
//

#include "curl/curl.h"
#include <iostream>
#include <vector>
using namespace std;
#include <opencv2/opencv.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include <stdlib.h>

using namespace cv;

#define streamip "http://192.168.1.2:8080/"
#define test1 "/Users/naveenmysore/Documents/test1.jpg"
#define test2 "/Users/naveenmysore/Documents/test2.jpg"
#define harr "/Users/naveenmysore/Documents/dronemov/haarcascade_frontalface_alt2.xml"
#define harr_upper "/Users/naveenmysore/Documents/dronemov/haarcascade_upperbody.xml"
#define testface "/Users/naveenmysore/Documents/dronemov/testface.jpg"
//#define harr "http://pngimg.com/upload/face_PNG11759.png"
//#define streamip "http://pngimg.com/upload/small/parrot_PNG722.png"


// curl section
size_t write_data(char *ptr, size_t size, size_t nmemb, void *userdata)
{
    vector<uchar> *stream = (vector<uchar>*)userdata;
    size_t count = size * nmemb;
    stream->insert(stream->end(), ptr, ptr + count);
    return count;
}

//function to retrieve the image as cv::Mat data type
cv::Mat curlImg(const char *img_url, int timeout=10)
{
    vector<uchar> stream;
    CURL *curl = curl_easy_init();
    curl_easy_setopt(curl, CURLOPT_URL, img_url); //the img url
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data); // pass the writefunction
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &stream); // pass the stream ptr to the writefunction
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeout); // timeout if curl_easy hangs,
    curl_easy_perform(curl); // start curl
    curl_easy_cleanup(curl); // cleanup
    return imdecode(stream, -1); // 'keep-as-is'
}

void display_image(string title, const cv::Mat &img){
    namedWindow(title, WINDOW_AUTOSIZE);
    imshow(title, img);
}

void desp_mat(string name, Mat M){
    cout<<name<<":"<<"size:"<<M.rows<<"X"<<M.cols<<" type:"<<M.type()<<endl;
}


void DrawLine( Mat img, Point start, Point end )
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


Mat hough_transform(Mat img){
    Mat frame = img.clone();
    //cvtColor( frame, frame, CV_BGR2GRAY );
    //GaussianBlur( frame, frame, Size(3,3), 0, 0, BORDER_DEFAULT );
    //cv::Canny(frame,frame,10,350);
    vector<Vec2f> lines;
    HoughLines(frame, lines, 1, CV_PI/180, 150, 0, 0 );
    for( size_t i = 0; i < lines.size(); i++ )
    {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        pt1.x = cvRound(x0 + 1000*(-b));
        pt1.y = cvRound(y0 + 1000*(a));
        pt2.x = cvRound(x0 - 1000*(-b));
        pt2.y = cvRound(y0 - 1000*(a));
        line( frame, pt1, pt2, Scalar(0,0,255), 3, CV_AA);
    }
    return frame;
}

Mat find_contours(Mat img) {
    Mat frame = img.clone();
    cvtColor( frame, frame, CV_BGR2GRAY );
    cv::Canny(frame,frame,10,350);
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours( frame, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
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
    
    Mat drawing = Mat::zeros( frame.size(), frame.type() );
    for( int i = 0; i< contours.size(); i++ )
    {
        Scalar color = Scalar( 255, 255, 255 );
        drawContours( drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
        //cout<< "br"<<boundRect[i].tl()<<endl;
        
        rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
        // circle( drawing, center[i], (int)radius[i], color, 2, 8, 0 );
    }
    uchar fillValue = 128;
    cv::Point seed(4,4);
    cv::Mat mask;
    cv::Canny(drawing, mask, 100, 200);
    cv::copyMakeBorder(mask, mask, 1, 1, 1, 1, cv::BORDER_REPLICATE);
    cv::floodFill(drawing, mask, seed, cv::Scalar(255) ,0, cv::Scalar(), cv::Scalar(), 4 | cv::FLOODFILL_MASK_ONLY | (fillValue << 8));
    return drawing;
}

 Rect max_contours(Mat img) {
    Mat frame = img.clone();
    cvtColor( frame, frame, CV_BGR2GRAY );
    cv::Canny(frame,frame,10,350);
    vector<vector<Point> > contours;
    vector<Vec4i> hierarchy;
    findContours( frame, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
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
    double temp = 0;
    int idx =0;
    for( int i = 0; i< contours.size(); i++ )
    {
        double area =contourArea( contours[i],false);
        if(area > temp){
            temp = area;
            idx = i;
        }
        
        // circle( drawing, center[i], (int)radius[i], color, 2, 8, 0 );
    }
    //rectangle( drawing, boundRect[idx].tl(), boundRect[idx].br(), color, 2, 8, 0 );
    Mat drawing = Mat::zeros( frame.size(), frame.type() );
    Scalar color = Scalar( 255, 255, 255 );
    rectangle( drawing, boundRect[idx].tl(), boundRect[idx].br(), color, 2, 8, 0 );
        // circle( drawing, center[i], (int)radius[i], color, 2, 8, 0 );
    return Rect(boundRect[idx].tl(),boundRect[idx].br());
}

Mat HarrTranform(Mat img){
    std::vector<Rect> faces;
    Mat frame = img.clone();
    CascadeClassifier face_cascade;
    face_cascade.load(harr);
    face_cascade.detectMultiScale( frame, faces);
    // Draw circles on the detected faces
    for( int i = 0; i < faces.size(); i++ )
    {
        Point center( faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5 );
        ellipse( frame, center, Size( faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar( 255, 0, 255 ), 4, 8, 0 );
        //cvRectangle(original,cvPoint(100,50),cvPoint(200,200),CV_RGB(255,0,0),5,8);
    }
    cvtColor(frame, frame, CV_BGR2GRAY );
    return frame;
}

Mat HarrisCorner(Mat img){
    Mat frame = img.clone();
    int thresh = 200;
    //cvtColor( frame, frame, CV_BGR2GRAY );
    //GaussianBlur( frame, frame, Size(3,3), 0, 0, BORDER_DEFAULT );
    //cv::Canny(frame,frame,10,350);
    Mat frame_copy = frame.clone();
    cornerHarris(frame, frame, 2, 3, 0.04, BORDER_DEFAULT );
    /// Normalizing
    normalize( frame, frame, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
    convertScaleAbs( frame, frame );
    /// Drawing a circle around corners
    for( int j = 0; j < frame.rows ; j++ ){
        for( int i = 0; i < frame.cols; i++ ){
            if( (int) frame.at<float>(j,i) > thresh ) {
                    circle( frame_copy, Point( i, j ), 1,  Scalar(255), 2, 8, 0 );
            }
        }
    }
    return frame_copy;
}

void FindDifference(cv::Mat src1, cv::Mat src2, cv::Mat &dst, int threshold) {
    dst = cv::abs(src2 - src1);
    cv::threshold(dst, dst, threshold, 255, cv::THRESH_BINARY);
}


int main(int argc, const char * argv[]) {
    
    Mat w1;
    Mat w2;
    Mat dst;
    Mat frame;
    vector<Point2f> features;
    vector<Point2f> track;
    vector<uchar> status;
    vector<float> err;
    cout<<"enter"<<endl;
    Size subPixWinSize(10,10), winSize(31,31);
    Point2f point;
    point = Point2f((float)100.0, (float)120.0);
    vector<Point2f> points[2];
    
    Mat frame1 = imread(test1, CV_LOAD_IMAGE_COLOR);
    Mat frame2 = imread(test2, CV_LOAD_IMAGE_COLOR);
    Mat f1 = frame1.clone();
    Mat f2 = frame2.clone();
    cvtColor(frame1, frame1, COLOR_BGR2GRAY);
    cvtColor(frame2, frame2, COLOR_BGR2GRAY);
    Mat ts;
    //FindDifference(f1, f2, ts, 100);
    //frame2 = frame2( max_contours(ts) );
    goodFeaturesToTrack(frame1,features,100,0.01,0.01, frame2,3,false, 0.04);//extract features to track
    cout<<features.size()<<endl;
    for(int i=0; i<features.size(); i++){
        cout<<features[i]<<endl;
    }
    calcOpticalFlowPyrLK(frame1,frame2,features,track,status,err);//optical flow
    for(int i=0; i<features.size(); i++){
        //cout<<features[i]<<"::"<<track[i]<<endl;
        DrawLine(frame1, features[i], track[i]);
    }

    display_image("source", frame1);
    //display_image("source2", ts);
    waitKey(0);
    return 0;
}