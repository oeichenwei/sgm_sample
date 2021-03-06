
#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>
#include <chrono>
#include "opencv2/opencv.hpp"

#include "sgbm.h"


void recv_file_path(std::string &path) {

  do {
    std::cin >> path;
    std::ifstream ifs(path); 

    if (ifs.is_open()) {
      break;
    }

    std::cout << "The specified file can not be opened. Please enter again. " << std::endl;

  } while (true);
  
}

int recv_int() {

  int val = 0;
  std::string val_str;
  do {

    try {
      std::cin >> val_str;
      val = std::stoi(val_str);
      break;
    } catch (std::exception e) {
      std::cout << "Input data is not integer. Please enter integer." << std::endl;
    }

  } while(true);

  return val;
}

void recv_console_input(std::string &left_path, std::string &right_path, int &disp_range, int &p1, int &p2) {

  std::cout << "Please enter left image path." << std::endl;
  recv_file_path(left_path);

  std::cout << "Please enter right image path." << std::endl;
  recv_file_path(right_path);

  std::cout << "Please specify disparity range." << std::endl;
  disp_range = recv_int();

  std::cout << "Please specify p1." << std::endl;
  p1 = recv_int();

  std::cout << "Please specify p2." << std::endl;
  p2 = recv_int();

  return;
}

int main(int argc, char** argv)
{
    std::cout << "SGBM Test Started!" << std::endl;
    
    std::string left_path, right_path;
    
    left_path = "/Users/wilsonc/works/LOGApp/samples/sgbm/top-left.png";
    right_path = "/Users/wilsonc/works/LOGApp/samples/sgbm/top-right.png";
    
    std::cout << "0. Load parameters from user." << std::endl;
    //recv_console_input(left_path, right_path, disp_r, p1, p2);
    
    std::cout << "1. Open and load images" << std::endl;
    cv::Mat left = cv::imread(left_path, cv::IMREAD_GRAYSCALE);
    cv::Mat right = cv::imread(right_path, cv::IMREAD_GRAYSCALE);
//    cv::resize(left, left, cv::Size(left.cols / 2, left.rows / 2));
//    cv::resize(right, right, cv::Size(right.cols / 2, right.rows / 2));

    std::cout << "2. Initialize class" << std::endl;
    
    int disp_r = 256;//, p1, p2;
    Sgbm sgbm(left.rows, left.cols, disp_r, 3, 20, true, false);
    cv::Mat disp;
    
    auto begin = std::chrono::system_clock::now();
    for (int i = 0; i < 1; i++)
    {
#if 1
    sgbm.compute_disp(left, right, disp);
#else
    int NumDisparities = 256;
    StereoSGBMParams params(0, NumDisparities, 8, 8 * 64, 32 * 64, 1, 63, 5, 100, 10, cv::StereoSGBM::MODE_SGBM);
    cv::Mat disp16(left.size(), CV_16UC1), buffer;
    computeDisparitySGBM(left, right, disp16, params, buffer);
    disp16.convertTo(disp, CV_8U, 255 / (NumDisparities * 16.));
#endif
    }
    auto end = std::chrono::system_clock::now();
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    std::cout << "3. costed time=" << milliseconds.count() << std::endl;

    cv::imshow("Sgbm Result", disp);
    cv::waitKey(0);
    return 0;
}
