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
#include <stdlib.h>

using namespace cv;

#define streamip "http://192.168.1.2:8080/"
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
    CURLcode res = curl_easy_perform(curl); // start curl
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

int main(int argc, const char * argv[]) {
    
    //VideoCapture cap;
    Mat frame;
    //frame = imread(streamip, CV_LOAD_IMAGE_COLOR);
    cout<<"enter"<<endl;
    while(true){
    frame = curlImg(streamip);
    //cout<<frame.size()<<endl;
    if (frame.empty()) {
        cout << "failed to load image"<<endl;
        return -1; // load fail
    } else {
        //imshow("Stream from Drone.", frame);
        display_image("Stream from Drone.", frame);
        waitKey(20);
    }
    }
    waitKey(0);
    return 0;
}
