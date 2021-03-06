
#include "opencv2/opencv.hpp"
#include <memory>

template <class AGGR_DATA_TYPE>
class cost_3d_array
{
public:
    cost_3d_array() : rows_(0), cols_(0), drange_(0), data_(nullptr) {
    }
    
    void reset(int rows, int cols, int drange) {
        if (data_)
            delete [] data_;
        
        rows_ = rows;
        cols_ = cols;
        drange_ = drange;
        
        int dataLen = rows * cols * drange * sizeof(AGGR_DATA_TYPE);
        data_raw_ = new uint8_t[dataLen + 31];
        data_ = (AGGR_DATA_TYPE*)((intptr_t)data_raw_ + 31 & ~intptr_t((void*)31));
        memset(data_, 0, sizeof(AGGR_DATA_TYPE) * rows * cols * drange);
    }
    
    virtual ~cost_3d_array() {
        if (data_raw_)
            delete [] data_raw_;
        data_raw_ =  nullptr;
        data_ = nullptr;
    }
    
    AGGR_DATA_TYPE& data(int row, int col, int d) {
        return data_[(cols_ * row + col) * drange_ + d];
    }
    
    AGGR_DATA_TYPE* ptr(int row, int col) const {
        return &(data_[(cols_ * row + col) * drange_]);
    }

protected:
    int rows_;
    int cols_;
    int drange_;
    
    uint8_t *data_raw_;
    AGGR_DATA_TYPE *data_;
};

typedef cost_3d_array<uint16_t>* cost_4d_array;

class ScanLine {
public:
  ScanLine(int drow, int dcol, bool posdir) {
    this->drow = drow;
    this->dcol = dcol;
    this->posdir = posdir;
  }
  bool posdir;
  int drow, dcol;
};

class ScanLines8 {
public:
  ScanLines8() {
    this->path8.push_back(ScanLine(0, 1, true));
    this->path8.push_back(ScanLine(1, 0, true));
    this->path8.push_back(ScanLine(0, -1, false));
    this->path8.push_back(ScanLine(-1, 0, false));
//    this->path8.push_back(ScanLine(1, 1, true));
//    this->path8.push_back(ScanLine(1, -1, true));
//    this->path8.push_back(ScanLine(-1, -1, false));
//    this->path8.push_back(ScanLine(-1, 1, false));
  }
  
  std::vector<ScanLine> path8;
};

class Sgbm {

public:
  Sgbm(int rows, int cols, int d_range, unsigned short p1, 
  unsigned short p2, bool gauss_filt = false, bool show_res = false);

  ~Sgbm();

  void reset_buffer();

  void compute_disp(cv::Mat &left, cv::Mat &right, cv::Mat &disp);

  void aggregate_cost(int row, int col, bool isEdge, uint8_t* p_agg, uint8_t& min_agg, bool initsum);

  void aggregate_cost_for_each_scanline();

  void calc_disparity(cv::Mat &disp_img);

  void census_transform(cv::Mat &img, cv::Mat &census);

  void calc_pixel_cost();

  unsigned char calc_hamming_dist(unsigned char val_l, unsigned char val_r);

public:

  bool gauss_filt, show_res;
  int rows, cols, d_range, scanpath;
  unsigned short p1, p2;
  cv::Mat census_l, census_r, disp_img;
  cost_3d_array<uint8_t> pix_cost;
  cost_3d_array<uint8_t> sum_cost;
  ScanLines8 scanlines;
    
protected:
  bool useSIMD_;
};


struct StereoSGBMParams
{
    StereoSGBMParams()
    {
        minDisparity = numDisparities = 0;
        SADWindowSize = 0;
        P1 = P2 = 0;
        disp12MaxDiff = 0;
        preFilterCap = 0;
        uniquenessRatio = 0;
        speckleWindowSize = 0;
        speckleRange = 0;
        mode = cv::StereoSGBM::MODE_SGBM;
    }
    
    StereoSGBMParams( int _minDisparity, int _numDisparities, int _SADWindowSize,
                     int _P1, int _P2, int _disp12MaxDiff, int _preFilterCap,
                     int _uniquenessRatio, int _speckleWindowSize, int _speckleRange,
                     int _mode )
    {
        minDisparity = _minDisparity;
        numDisparities = _numDisparities;
        SADWindowSize = _SADWindowSize;
        P1 = _P1;
        P2 = _P2;
        disp12MaxDiff = _disp12MaxDiff;
        preFilterCap = _preFilterCap;
        uniquenessRatio = _uniquenessRatio;
        speckleWindowSize = _speckleWindowSize;
        speckleRange = _speckleRange;
        mode = _mode;
    }
    
    int minDisparity;
    int numDisparities;
    int SADWindowSize;
    int preFilterCap;
    int uniquenessRatio;
    int P1;
    int P2;
    int speckleWindowSize;
    int speckleRange;
    int disp12MaxDiff;
    int mode;
};

void computeDisparitySGBM(const cv::Mat& img1, const cv::Mat& img2,
                                 cv::Mat& disp1, const StereoSGBMParams& params,
                                 cv::Mat& buffer );
