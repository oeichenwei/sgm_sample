
#include <iostream>
#include <string>
#include "opencv2/opencv.hpp"

#include "sgbm.h"

Sgbm::Sgbm(int rows, int cols, int d_range, unsigned short p1, 
  unsigned short p2, bool gauss_filt, bool show_res) {

  this->rows = rows;
  this->cols = cols;
  this->d_range = d_range;
  this->census_l = new cv::Mat(rows, cols, CV_8UC1);
  this->census_r = new cv::Mat(rows, cols, CV_8UC1);
  this->disp_img = new cv::Mat(rows, cols, CV_8UC1);
  this->p1 = p1;
  this->p2 = p2;
  this->scanpath = 8;
  this->gauss_filt = gauss_filt;
  this->show_res = show_res;

  reset_buffer();
}

Sgbm::~Sgbm() 
{
  delete this->census_l;
  delete this->census_r;
}


void Sgbm::compute_disp(cv::Mat &left, cv::Mat &right, cv::Mat &disp)
{

  // 0. Reset Buffer.
  reset_buffer();

  if (this->gauss_filt) {
    cv::GaussianBlur(left, left, cv::Size(3, 3), 3);
    cv::GaussianBlur(right, right, cv::Size(3, 3), 3);
  }

  if (this->show_res) {
    cv::imshow("Left Original", left);
    cv::imshow("Right Original", right);
    cv::waitKey(0);
  }

  // 1. Census Transform.
  census_transform(left, *this->census_l);
  census_transform(right, *this->census_r);

  if (this->show_res) {
    cv::imshow("Census Trans Left", *this->census_l);
    cv::imshow("Census Trans Right", *this->census_r);
    cv::waitKey(0);
  }

  // 2. Calculate Pixel Cost.
  calc_pixel_cost(*this->census_l, *this->census_r, this->pix_cost);
  
  // 3. Aggregate Cost
  aggregate_cost_for_each_scanline(this->pix_cost, this->agg_cost, this->sum_cost);

  // 4. Create Disparity Image.
  calc_disparity(this->sum_cost, *this->disp_img);

  // Visualize Disparity Image.
  disp = *this->disp_img;
  if (this->show_res) {
    cv::Mat tmp;
    disp.convertTo(tmp, CV_8U, 256.0/this->d_range);
    cv::imshow("Sgbm Result", tmp);
    cv::waitKey(0);
  }

  return;
}

void Sgbm::reset_buffer() 
{

  *(census_l) = 0;
  *(census_r) = 0;

  // Resize vector for Pix Cost
  pix_cost.reset(rows, cols, d_range);
  sum_cost.reset(rows, cols, d_range);

  // Resize vector for Agg Cost
  agg_cost.resize(scanpath);
  agg_min.resize(scanpath);
  for (int path = 0; path < scanpath; path++) {
    agg_cost[path].reset(rows, cols, d_range);
    agg_min[path] = cv::Mat::zeros(rows, cols, CV_16UC1);
  }
}

void Sgbm::census_transform(cv::Mat &img, cv::Mat &census)
{
  unsigned char * const img_pnt_st = img.data;
  unsigned char * const census_pnt_st = census.data;

  for (int row=1; row<rows-1; row++) {
    for (int col=1; col<cols-1; col++) {

      unsigned char *center_pnt = img_pnt_st + cols*row + col;
      unsigned char val = 0;
      for (int drow=-1; drow<=1; drow++) {
        for (int dcol=-1; dcol<=1; dcol++) {
          
          if (drow == 0 && dcol == 0) {
            continue;
          }
          unsigned char tmp = *(center_pnt + dcol + drow*cols);
          val = (val + (tmp < *center_pnt ? 0 : 1)) << 1;        
        }
      }
      *(census_pnt_st + cols*row + col) = val;
    }
  }
  return;
}

void Sgbm::calc_pixel_cost(cv::Mat &census_l, cv::Mat &census_r, cost_3d_array &pix_cost) {

  unsigned char * const census_l_ptr_st = census_l.data;
  unsigned char * const census_r_ptr_st = census_r.data;

  for (int row = 0; row < this->rows; row++) {
    for (int col = 0; col < this->cols; col++) {
      unsigned char val_l = static_cast<unsigned char>(*(census_l_ptr_st + row*cols + col));
      for (int d = 0; d < this->d_range; d++) {
        unsigned char val_r = 0;
        if (col - d >= 0) {
          val_r = static_cast<unsigned char>(*(census_r_ptr_st + row*cols + col - d));
        }
        pix_cost.data(row, col, d) = calc_hamming_dist(val_l, val_r);
      }
    }
  }
}

/// https://www.tutorialspoint.com/what-is-hamming-distance
unsigned char Sgbm::calc_hamming_dist(unsigned char val_l, unsigned char val_r) {

  unsigned char dist = 0;
  unsigned char d = val_l ^ val_r;

  while(d) {
    d = d & (d - 1);
    dist++;
  }
  return dist;  
}

unsigned short Sgbm::aggregate_cost(int row, int col, int depth, int path, cost_3d_array &pix_cost, cost_4d_array &agg_cost) {

  // Depth loop for current pix.
  unsigned long val0 = 0xFFFF;
  unsigned long val1 = 0xFFFF;
  unsigned long val2 = 0xFFFF;
  unsigned long val3 = 0xFFFF;
  unsigned long min_prev_d = 0xFFFF;

  int dcol = this->scanlines.path8[path].dcol;
  int drow = this->scanlines.path8[path].drow;

  // Pixel matching cost for current pix.
  unsigned long indiv_cost = pix_cost.data(row, col, depth);

  if (row - drow < 0 || this->rows <= row - drow || col - dcol < 0 || this->cols <= col - dcol) {
    agg_cost[path].data(row, col, depth) = indiv_cost;
    return agg_cost[path].data(row, col, depth);
  }

    min_prev_d = agg_min[path].at<uint16_t>(row - drow, col - dcol);
    val0 = agg_cost[path].data(row - drow, col - dcol, depth);
    val1 = agg_cost[path].data(row - drow, col - dcol, depth - 1) + p1;
    val2 = agg_cost[path].data(row - drow, col - dcol, depth + 1) + p1;
    val3 = min_prev_d + p2;
/*  // Depth loop for previous pix.
  for (int dd = 0; dd < this->d_range; dd++) {
    unsigned long prev = agg_cost[path].data(row-drow, col-dcol, dd);
    if (prev < min_prev_d) {
      min_prev_d = prev;
    }
    
    if (depth == dd) {
      val0 = prev;
    } else if (depth == dd + 1) {
      val1 = prev + this->p1;
    } else if (depth == dd - 1) {
      val2 = prev + this->p1;
    } else {
      unsigned long tmp = prev + this->p2;
      if (tmp < val3) {
        val3 = tmp;
      }            
    }
  }
*/
  // Select minimum cost for current pix.
  agg_cost[path].data(row, col, depth) = std::min(std::min(std::min(val0, val1), val2), val3) + indiv_cost - min_prev_d;
  //agg_cost[path][row][col][depth] = indiv_cost;

  return agg_cost[path].data(row, col, depth);
}

void Sgbm::aggregate_cost_for_each_scanline(cost_3d_array &pix_cost, cost_4d_array &agg_cost, cost_3d_array &sum_cost)
{
  // Cost aggregation for positive direction.
  for (int path = 0; path < this->scanlines.path8.size(); path++) {
    for (int row = 0; row < this->rows; row++) {
      for (int col = 0; col < this->cols; col++) {
        if (this->scanlines.path8[path].posdir) {
          //std::cout << "Pos : " << path << std::endl;
          uint16_t& minAgg = agg_min[path].at<uint16_t>(row, col);
          minAgg = 0xFFFF;
          for (int d = 0; d < this->d_range; d++) {
            unsigned short a_cost = aggregate_cost(row, col, d, path, pix_cost, agg_cost);
            if (a_cost < minAgg)
                minAgg = a_cost;
            sum_cost.data(row, col, d) += a_cost;
          }
        }
      }
    }
  }

  // Cost aggregation for negative direction.
  for (int path = 0; path < this->scanlines.path8.size(); path++) {
    for (int row = this->rows - 1; 0 <= row; row--) {
      for (int col = this->cols - 1; 0 <= col; col--) {
        if (!this->scanlines.path8[path].posdir) {
          //std::cout << "Neg : " << path << std::endl;
          uint16_t& minAgg = agg_min[path].at<uint16_t>(row, col);
          minAgg = 0xFFFF;
          for (int d = 0; d < this->d_range; d++) {
              unsigned short a_cost = aggregate_cost(row, col, d, path, pix_cost, agg_cost);
              if (a_cost < minAgg)
                  minAgg = a_cost;
              sum_cost.data(row, col, d) += a_cost;
          }
        }
      }
    }
  }
  return;
}

void Sgbm::calc_disparity(cost_3d_array &sum_cost, cv::Mat &disp_img)
{
  for (int row = 0; row < this->rows; row++) {
    for (int col = 0; col < this->cols; col++) {
      unsigned char min_depth = 0;
      unsigned long min_cost = sum_cost.data(row, col, min_depth);
      for (int d = 1; d < this->d_range; d++) {
        unsigned long tmp_cost = sum_cost.data(row, col, d);
        if (tmp_cost < min_cost) {
          min_cost = tmp_cost;
          min_depth = d;
        }
      }
      disp_img.at<unsigned char>(row, col) = min_depth;
    } 
  } 

  return;
}

