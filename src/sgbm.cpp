
#include <iostream>
#include <string>
#include <chrono>
#include "opencv2/opencv.hpp"
#include "precomp.hpp"

#include "sgbm.h"

#define SGBM_MIN(x, y)  (((x) < (y)) ? (x) : (y))
#define SGBM_MAX(x, y)  (((x) > (y)) ? (x) : (y))

Sgbm::Sgbm(int rows, int cols, int d_range, unsigned short p1,
           unsigned short p2, bool gauss_filt, bool show_res)
{
    this->rows = rows;
    this->cols = cols;
    this->d_range = d_range;
    this->census_l = cv::Mat::zeros(rows, cols + d_range, CV_8UC1);
    this->census_r = cv::Mat::zeros(rows, cols + d_range, CV_8UC1);
    this->disp_img = cv::Mat::zeros(rows, cols, CV_8UC1);
    this->p1 = p1;
    this->p2 = p2;
    this->scanpath = scanlines.path8.size();
    this->gauss_filt = gauss_filt;
    this->show_res = show_res;
    
#if CV_SIMD128
    useSIMD_ = cv::hasSIMD128();
#endif

    reset_buffer();
}

Sgbm::~Sgbm() 
{
}

void Sgbm::compute_disp(cv::Mat &leftin, cv::Mat &rightin, cv::Mat &disp)
{
    cv::Mat left, right;
    if (this->gauss_filt) {
        cv::GaussianBlur(leftin, left, cv::Size(3, 3), 3);
        cv::GaussianBlur(rightin, right, cv::Size(3, 3), 3);
    }
    else {
        left = leftin;
        right = rightin;
    }
    
    if (this->show_res) {
        cv::imshow("Left Original", left);
        cv::imshow("Right Original", right);
        cv::waitKey(10);
    }
    
    auto begin = std::chrono::system_clock::now();
    // 1. Census Transform.
    census_transform(left, census_l);
    census_transform(right, census_r);
    auto end = std::chrono::system_clock::now();
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    std::cout << "census compute time=" << milliseconds.count() << std::endl;

    if (this->show_res) {
        cv::imshow("Census Trans Left", census_l);
        cv::imshow("Census Trans Right", census_r);
        cv::waitKey(10);
    }
    
    // 2. Calculate Pixel Cost.
    calc_pixel_cost();
    
    end = std::chrono::system_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    std::cout << "cost compute time=" << milliseconds.count() << std::endl;

    // 3. Aggregate Cost
    aggregate_cost_for_each_scanline();
    
    end = std::chrono::system_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    std::cout << "aggregate compute time=" << milliseconds.count() << std::endl;
    
    // 4. Create Disparity Image.
    calc_disparity(disp_img);

    end = std::chrono::system_clock::now();
    milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
    std::cout << "calc disparity compute time=" << milliseconds.count() << std::endl;

    // Visualize Disparity Image.
    disp = disp_img;
    if (this->show_res) {
        cv::Mat tmp;
        disp.convertTo(tmp, CV_8U, 256.0/this->d_range);
    }
}

void Sgbm::reset_buffer() 
{
    // Resize vector for Pix Cost
    pix_cost.reset(rows, cols, d_range);
    sum_cost.reset(rows, cols, d_range);
}

void Sgbm::census_transform(cv::Mat &img, cv::Mat &census)
{
    uint8_t * const img_pnt_st = img.data;
    uint8_t * const census_pnt_st = census.data;
#if CV_SIMD128
    cv::v_uint16x8 one = cv::v_setall_u16(1);
#endif
    for (int row = 1; row < rows - 1; row++) {
        int colstart = 1;
        uint8_t *center_pnt_row = img_pnt_st + cols * row + colstart;
        uint8_t *census_pnt_row = census_pnt_st + census.cols * row + colstart + d_range;
#if CV_SIMD128
        if(useSIMD_)
        {
            colstart = 8 * ((cols - 2) / 8) + colstart;
            for (int col = 1; col < colstart; col += 8, center_pnt_row += 8, census_pnt_row += 8) {
                cv::v_uint16x8 val = cv::v_setzero_u16();
                cv::v_uint16x8 ref = cv::v_load_expand(center_pnt_row);
                for (int drow = -1; drow <= 1; drow++) {
                    for (int dcol = -1; dcol <= 1; dcol++) {
                        if (drow == 0 && dcol == 0) {
                            continue;
                        }
                        cv::v_uint16x8 test = cv::v_load_expand(center_pnt_row + dcol + drow * cols);
                        cv::v_uint16x8 cmp = cv::v_min(one, test >= ref);
                        val = val << 1;
                        val = (val + cmp);
                    }
                }
                cv::v_pack_store(census_pnt_row, val);
            }
        }
#endif
        for (int col = colstart; col < cols - 1; col++, center_pnt_row++, census_pnt_row++) {
            unsigned char val = 0;
            for (int drow = -1; drow <= 1; drow++) {
                for (int dcol = -1; dcol <= 1; dcol++) {
                    if (drow == 0 && dcol == 0) {
                        continue;
                    }
                    unsigned char tmp = *(center_pnt_row + dcol + drow * cols);
                    val = val << 1;
                    val = (val + (tmp < *center_pnt_row ? 0 : 1));
                }
            }

            *census_pnt_row = val;
        }
    }
    return;
}

void Sgbm::calc_pixel_cost()
{
    uint8_t * const census_l_ptr_st = census_l.data;
    uint8_t * const census_r_ptr_st = census_r.data;

    for (int row = 0; row < rows; row++) {
        uint8_t *census_l_row_ptr = census_l_ptr_st + row * census_l.cols + d_range;
        uint8_t *census_r_row_ptr = census_r_ptr_st + row * census_r.cols + d_range;
        for (int col = 0; col < cols; col++, census_l_row_ptr++) {
            uint8_t val_l = *census_l_row_ptr;
            uint8_t* dest = pix_cost.ptr(row, col);
#if CV_SIMD128
            if(useSIMD_)
            {
                cv::v_uint8x16 ref = cv::v_setall_u8(val_l);
                for (int d = 0; d < d_range; d += 16, dest += 16) {
                    cv::v_uint8x16 test = cv::v_load(census_r_row_ptr + col - (d + 16));
                    cv::v_uint8x16 dist = ref ^ test;
                    
                    cv::v_uint16x8 dist16 = cv::v_reinterpret_as_u16(dist);
                    const cv::v_uint16x8 const77 = cv::v_setall_u16(0x7777);
                    const cv::v_uint16x8 const33 = cv::v_setall_u16(0x3333);
                    const cv::v_uint16x8 const11 = cv::v_setall_u16(0x1111);
                    const cv::v_uint16x8 constFF = cv::v_setall_u16(0x0F0F);
                    cv::v_uint16x8 dist16_7 = (dist16 >> 1) & const77;
                    cv::v_uint16x8 dist16_3 = (dist16 >> 2) & const33;
                    cv::v_uint16x8 dist16_1 = (dist16 >> 3) & const11;
                    dist16 = dist16 - dist16_7 - dist16_3 - dist16_1;
                    cv::v_uint16x8 result16 = ((dist16 >> 4) + dist16) & constFF;

                    uint8_t CV_DECL_ALIGNED(32) tmp[16];
                    cv::v_store_aligned(tmp, cv::v_reinterpret_as_u8(result16));
                    //v_reverse needs SSE3 or NEON, that can be 1x fast
                    for (int i = 0; i < 16; i++)
                        dest[i] = tmp[15 - i];
                }
            }
            else
#endif
            {
                for (int d = 0; d < d_range; d++) {
                    uint8_t val_r = 0;
                    if (col - d >= 0) {
                        val_r = *(census_r_row_ptr + col - d);
                    }
                    dest[d] = calc_hamming_dist(val_l, val_r);
                }
            }
        }
    }
}

/// https://www.tutorialspoint.com/what-is-hamming-distance
unsigned char Sgbm::calc_hamming_dist(unsigned char val_l, unsigned char val_r)
{
    unsigned char dist = 0;
    unsigned char d = val_l ^ val_r;
    
    while(d) {
        d = d & (d - 1);
        dist++;
    }
    return dist;
}

void Sgbm::aggregate_cost(int row, int col, bool isEdge, uint8_t* p_val, uint8_t& min_prev_d, bool initsum)
{
    uint8_t minAgg = 0xFFFF;
    uint8_t* p_cost = pix_cost.ptr(row, col);
    
    uint8_t CV_DECL_ALIGNED(32) new_agg[d_range];
    uint8_t* p_agg = new_agg;
    
    uint8_t* p_sum = sum_cost.ptr(row, col);

#if CV_SIMD128
    if(useSIMD_)
    {
        cv::v_uint8x16 val3 = cv::v_setall_u8(min_prev_d + p2);
        const cv::v_uint8x16 vp1 = cv::v_setall_u8(p1);
        const cv::v_uint8x16 vmin_prev = cv::v_setall_u8(min_prev_d);
        for (int depth = 0; depth < d_range; depth += 16, p_agg += 16, p_sum += 16, p_cost += 16) {
            cv::v_uint8x16 retval;
            cv::v_uint8x16 indiv_cost = cv::v_load_aligned(p_cost);
            if (isEdge) {
                retval = indiv_cost;
            } else {
                retval = cv::v_setzero_u8();
                const cv::v_uint8x16 val0 = cv::v_load_aligned(p_val + depth);
                cv::v_uint8x16 val1, val2;
                val1 = cv::v_load(p_val + depth - 1) + vp1;
                val2 = cv::v_load(p_val + depth + 1) + vp1;
                
                cv::v_uint8x16 val = cv::v_min(val0, val1);
                val = cv::v_min(val, val2);
                val = cv::v_min(val, val3);
                retval = val + indiv_cost - vmin_prev;
            }
            cv::v_uint16x8 retlow, rethigh;
            cv::v_expand(retval, retlow, rethigh);
            uint16_t lmin = cv::v_reduce_min(retlow);
            uint16_t hmin = cv::v_reduce_min(rethigh);

            lmin = lmin < hmin ? lmin : hmin;
            if (lmin < minAgg)
                minAgg = lmin;

            cv::v_store_aligned(p_agg, retval);
            if (initsum) {
                cv::v_store_aligned(p_sum, retval);
            }
            else {
                cv::v_uint8x16 v_sum = cv::v_load_aligned(p_sum);
                v_sum = v_sum + retval;
                cv::v_store_aligned(p_sum, v_sum);
            }
        }
    }
    else
#endif
    {
        uint16_t val3 = min_prev_d + p2;
        for (int depth = 0; depth < d_range; depth++, p_agg++, p_sum++, p_cost++) {
            uint16_t retval = 0;
            uint16_t indiv_cost = *p_cost;
            if (isEdge) {
                retval = indiv_cost;
            }
            else {
                uint16_t val = p_val[depth];
                uint16_t val1;
                uint16_t val2;
                if (depth > 0) {
                    val1 = p_val[depth - 1] + p1;
                    if (val1 < val)
                        val = val1;
                }
                if (depth < d_range - 1) {
                    val2 = p_val[depth + 1] + p1;
                    if (val2 < val)
                        val = val2;
                }
                if (val3 < val)
                    val = val3;
                
                retval = val + indiv_cost - min_prev_d;
            }
            *p_agg = retval;
            if (retval < minAgg)
                minAgg = retval;
            if (initsum)
                *p_sum = retval;
            else
                *p_sum += retval;
        }
    }

    min_prev_d = minAgg;
    memcpy(p_val, new_agg, d_range * sizeof(uint8_t));
}

void Sgbm::aggregate_cost_for_each_scanline()
{
    bool is_edge = false;
    uint8_t CV_DECL_ALIGNED(32) last_agg[d_range + 64];
    uint8_t min_agg = 0xFFFF;
    uint8_t *offset_agg = last_agg + 32;
    memset(last_agg, 0xff, (d_range + 64) * sizeof(uint8_t));

    /// Cost aggregation for positive direction.
    /// left -> right
    for (int row = 0; row < rows; row++) {
        is_edge = (row == 0);
        for (int col = 0; col < cols; col++) {
            aggregate_cost(row, col, is_edge, offset_agg, min_agg, true);
        }
    }

    min_agg = 0xFFFF;
    /// top -> bottom
    for (int col = 0; col < cols; col++) {
        is_edge = (col == 0);
        for (int row = 0; row < rows; row++) {
            aggregate_cost(row, col, is_edge, offset_agg, min_agg, false);
        }
    }

    min_agg = 0xFFFF;
    /// right -> left
    for (int row = rows - 1; row >= 0; row--) {
        is_edge = (row == rows - 1);
        for (int col = cols - 1; col >= 0; col--) {
            aggregate_cost(row, col, is_edge, offset_agg, min_agg, false);
        }
    }

    min_agg = 0xFFFF;
    /// bottom -> left
    for (int col = cols - 1; col >= 0; col--) {
        is_edge = (col == cols - 1);
        for (int row = rows - 1; row >= 0; row--) {
            aggregate_cost(row, col, is_edge, offset_agg, min_agg, false);
        }
    }
}

void Sgbm::calc_disparity(cv::Mat &disp_img)
{
    for (int row = 0; row < this->rows; row++) {
        uint8_t* disp_row_ptr = disp_img.ptr(row);
        for (int col = 0; col < this->cols; col++) {
#if CV_SIMD128
            if(useSIMD_)
            {
                unsigned char min_depth = 0;
                unsigned long min_cost = 0xffff;
                uint8_t *p_sum = sum_cost.ptr(row, col);
                const cv::v_uint16x8 mask = cv::v_setall_u16(0x00FF);
                for (int d = 0; d < this->d_range; d += 16) {
                    cv::v_uint8x16 v_sum = cv::v_load_aligned(p_sum + d);
                    cv::v_uint16x8 v_sum_l = cv::v_reinterpret_as_u16(v_sum);
                    uint16_t l_min = cv::v_reduce_min(v_sum_l & mask);
                    uint16_t r_min = cv::v_reduce_min((v_sum_l >> 8) & mask);
                    
                    if (l_min < min_cost) {
                        min_depth = d;
                        min_cost = l_min;
                    }
                    if (r_min < min_cost) {
                        min_depth = d + 1;
                        min_cost = r_min;
                    }
                }
                for (int i = 0; i < 16; i += 2) {
                    if (p_sum[min_depth + i] == min_cost) {
                        min_depth += i;
                        break;
                    }
                }
                disp_row_ptr[col] = min_depth;
            }
            else
#endif
            {
                unsigned char min_depth = 0;
                unsigned long min_cost = sum_cost.data(row, col, min_depth);
                for (int d = 1; d < this->d_range; d++) {
                    unsigned long tmp_cost = sum_cost.data(row, col, d);
                    if (tmp_cost < min_cost) {
                        min_cost = tmp_cost;
                        min_depth = d;
                    }
                }
                disp_row_ptr[col] = min_depth;
            }
        }
    }
}
