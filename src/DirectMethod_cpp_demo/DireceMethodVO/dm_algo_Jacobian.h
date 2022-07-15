#ifndef DM_ALGO_H
#define DM_ALGO_H

// #pragma once

#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include <boost/format.hpp>
// #include <pangolin/pangolin.h>
#include <mutex>

using namespace std;

typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;

// // Camera intrinsics
// double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
// // baseline
// double baseline = 0.573;

// useful typedefs
typedef Eigen::Matrix<double, 6, 6> Matrix6d;
typedef Eigen::Matrix<double, 2, 6> Matrix26d;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

class dm_algo
{
public:
    dm_algo();
    ~dm_algo();

    cv::Mat Track(const cv::Mat &currentRGB, const cv::Mat &currentDepth);

private:
    cv::RNG mrng;

    int mnPoints = 2000;
    int mboarder = 30; // 20

    VecVector2d mpixels_ref;
    vector<double> mdepth_ref;

    cv::Mat mLastRGB;
    cv::Mat mLastDepth;
};

/// class for accumulator jacobians in parallel
class JacobianAccumulator
{
public:
    JacobianAccumulator(
        const cv::Mat &img1_,
        const cv::Mat &img2_,
        const VecVector2d &px_ref_,
        const vector<double> depth_ref_,
        Sophus::SE3d &T21_) : img1(img1_), img2(img2_),
                              px_ref(px_ref_), depth_ref(depth_ref_),
                              T21(T21_)
    {
        mpx_size = px_ref.size();

        // projection = VecVector2d(px_ref.size(), Eigen::Vector2d(0, 0));
        projection = VecVector2d(mpx_size, Eigen::Vector2d(0, 0));
    }

    /// accumulate jacobians in a range
    // void accumulate_jacobian(const cv::Range &range);
    void accumulate_jacobian();

    /// get hessian matrix
    Matrix6d hessian() const { return mH; }

    /// get bias
    Vector6d bias() const { return mb; }

    /// get total cost
    double cost_func() const { return mcost; }

    /// get projected points
    VecVector2d projected_points() const { return projection; }

    /// reset h, b, cost to zero
    void reset()
    {
        mH = Matrix6d::Zero();
        mb = Vector6d::Zero();
        mcost = 0;
    }

private:
    const cv::Mat &img1;
    const cv::Mat &img2;
    const VecVector2d &px_ref;
    const vector<double> depth_ref;
    Sophus::SE3d &T21;
    VecVector2d projection; // projected points

    int mpx_size;

    // std::mutex hessian_mutex;
    Matrix6d mH = Matrix6d::Zero();
    Vector6d mb = Vector6d::Zero();
    double mcost = 0;
};

void DirectPoseEstimationMultiLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    Sophus::SE3d &T21);

void DirectPoseEstimationSingleLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    Sophus::SE3d &T21);

// bilinear interpolation
inline float GetPixelValue(const cv::Mat &img, float x, float y)
{
    // boundary check
    if (x < 0)
        x = 0;
    if (y < 0)
        y = 0;
    if (x >= img.cols)
        x = img.cols - 1;
    if (y >= img.rows)
        y = img.rows - 1;
    uchar *data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
        (1 - xx) * (1 - yy) * data[0] +
        xx * (1 - yy) * data[1] +
        (1 - xx) * yy * data[img.step] +
        xx * yy * data[img.step + 1]);
}

#endif