#ifndef TRACKING_H
#define TRACKING_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

// #include "Viewer.h"
// #include "FrameDrawer.h"
// #include "Map.h"
// #include "LocalMapping.h"
// #include "LoopClosing.h"
// #include "Frame.h"
// #include "ORBVocabulary.h"
// #include "KeyFrameDatabase.h"
// #include "ORBextractor.h"
// #include "Initializer.h"
// #include "MapDrawer.h"
#include "System.h"

#include <mutex>

using namespace std;
using namespace g2o;

#define IR_WHITE_DOTS_THRESHOLD 180

namespace DirectMethod
{

    struct Measurement
    {
        Measurement(Eigen::Vector3d p, float g) : pos_world(p), grayscale(g) {}
        Eigen::Vector3d pos_world;
        float grayscale;
    };

    struct RelativePoseResult
    {
        RelativePoseResult(Eigen::Isometry3d Tcw, double tframe)
            : Tcw_(Tcw), tframe_(tframe) {}

        Eigen::Isometry3d Tcw_;
        double tframe_;
    };

    // class Viewer;
    // class FrameDrawer;
    // class Map;
    // class LocalMapping;
    // class LoopClosing;
    class System;

    class Tracking
    {

    public:
        Tracking(System *pSys, const string &strSettingPath);

        // Preprocess the input and call Track(). Extract features and performs stereo matching.
        cv::Mat GrabImageDirectMethod_RGBD(const cv::Mat &imRGB, const cv::Mat &imD, const double &timestamp);

    public:
        cv::Mat mImGray;
        cv::Mat mImDepth;
        double mtimestamp;
        cv::Mat mImLastGray;
        cv::Mat mImLastDepth;

        // // Lists used to recover the full camera trajectory at the end of the execution.
        // // Basically we store the reference keyframe for each frame and its relative transformation
        // list<cv::Mat> mlRelativeFramePoses;
        // list<double> mlFrameTimes;

    protected:
        // Main tracking function. It is independent of the input sensor.
        void Track();

        inline Eigen::Vector3d project2Dto3D(
            int x, int y, float d)
        {
            float zz = float(d);
            float xx = zz * (x - mcx) / mfx;
            float yy = zz * (y - mcy) / mfy;
            return Eigen::Vector3d(xx, yy, zz);
        }

        inline Eigen::Vector2d project3Dto2D(
            float x, float y, float z)
        {
            float u = mfx * x / z + mcx;
            float v = mfy * y / z + mcy;
            return Eigen::Vector2d(u, v);
        }

        // 直接法估计位姿
        // 输入：测量值（空间点的灰度），新的灰度图，相机内参；
        // 输出：相机位姿
        // 返回：true为成功，false失败
        bool poseEstimationDirect(
            // const vector<Measurement> &measurements,
            // cv::Mat *gray,
            // Eigen::Matrix3f &intrinsics,
            // Eigen::Isometry3d &Tcw
        );

        // System
        System *mpSystem;

        // Calibration matrix
        Eigen::Matrix3f mK;
        float mfx, mfy, mcx, mcy;
        cv::Mat mDistCoef;
        float mbf;

        // New KeyFrame rules (according to fps)
        int mMinFrames;
        int mMaxFrames;

        // For RGB-D inputs only. For some datasets (e.g. TUM) the depthmap values are scaled.
        float mDepthMapFactor;

        // Motion Model
        // cv::Mat mVelocity;

        // Color order (true RGB, false BGR, ignored if grayscale)
        bool mbRGB;

        //
        // directMethod related
        cv::RNG mrng;
        int mPoints;
        int mboarder;

        std::vector<Measurement> mmeasurements;
        std::vector<Measurement> mfirst_frame_measurements;
        bool mfirst_frame_flag;
        std::vector<RelativePoseResult> mvRelativePoseResult; // res

        Eigen::Isometry3d mT_curr_last;
        Eigen::Isometry3d mT_curr_world;
    };

} // namespace DirectMethod

// project a 3d point into an image plane, the error is photometric error
// an unary edge with one vertex SE3Expmap (the pose of camera)
// 边edge   误差项
// 顶点vertex 待优化的变量 pose
class EdgeSE3ProjectDirect : public BaseUnaryEdge<1, double, VertexSE3Expmap>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // 构造函数
    EdgeSE3ProjectDirect() {}

    EdgeSE3ProjectDirect(
        Eigen::Vector3d point,
        float fx, float fy, float cx, float cy,
        cv::Mat *image)
        : x_world_(point),
          fx_(fx), fy_(fy), cx_(cx), cy_(cy),
          image_(image)
    {
        // image_height_ = image_->cols;
        // image_width_ = image_->rows;
    }

    virtual void computeError();
    virtual void linearizeOplus();

    // dummy read and write functions because we don't care...
    virtual bool read(std::istream &in) {}
    virtual bool write(std::ostream &out) const {}

protected:
    // Bilinear interpolation
    // get a gray scale value from reference image
    inline float getPixelValue(float x, float y)
    {
        // // boundary check
        // if (x < 0)
        //     x = 0;
        // if (y < 0)
        //     y = 0;
        // if (x >= image_->cols)
        //     x = image_->cols - 1;
        // if (y >= image_->rows)
        //     y = image_->rows - 1;

        uchar *data = &image_->data[int(y) * image_->step + int(x)];
        float xx = x - floor(x);
        float yy = y - floor(y);
        return float(
            (1 - xx) * (1 - yy) * data[0] +      // z00 lower-left
            xx * (1 - yy) * data[1] +            // z10 lower-right
            (1 - xx) * yy * data[image_->step] + // z01 upper-left
            xx * yy * data[image_->step + 1]     // z11 upper-right
        );
    }

    inline float getPixelValue_opt(float x, float y,
                                   getPixelValueDirection opt)
    {
        // boundary check
        if (x < 0)
            x = 0;
        if (y < 0)
            y = 0;
        if (x >= image_->cols)
            x = image_->cols - 1;
        if (y >= image_->rows)
            y = image_->rows - 1;

        uchar *data = &image_->data[int(y) * image_->step + int(x)];

        bool isWhiteDot = (data[0] > IR_WHITE_DOTS_THRESHOLD) ||
                          (data[1] > IR_WHITE_DOTS_THRESHOLD) ||
                          (data[image_->step] > IR_WHITE_DOTS_THRESHOLD) ||
                          (data[image_->step + 1] > IR_WHITE_DOTS_THRESHOLD);
        while (isWhiteDot)
        {
            if (opt == u_plus)
            {
                if (x >= image_->cols - 1)
                    break;
                x++;
            }
            else if (opt == u_minus)
            {
                if (x < 1)
                    break;
                x--;
            }
            else if (opt == v_plus)
            {
                if (y >= image_->rows - 1)
                    break;
                y++;
            }
            else if (opt == v_minus)
            {
                if (y < 1)
                    break;
                y--;
            }
            data = &image_->data[int(y) * image_->step + int(x)];
            isWhiteDot = (data[0] > IR_WHITE_DOTS_THRESHOLD) ||
                         (data[1] > IR_WHITE_DOTS_THRESHOLD) ||
                         (data[image_->step] > IR_WHITE_DOTS_THRESHOLD) ||
                         (data[image_->step + 1] > IR_WHITE_DOTS_THRESHOLD);
        }

        float xx = x - floor(x);
        float yy = y - floor(y);
        return float(
            (1 - xx) * (1 - yy) * data[0] +      // z00 lower-left
            xx * (1 - yy) * data[1] +            // z10 lower-right
            (1 - xx) * yy * data[image_->step] + // z01 upper-left
            xx * yy * data[image_->step + 1]     // z11 upper-right
        );
    }

public:
    Eigen::Vector3d x_world_;                 // 3D point in world frame
    float cx_ = 0, cy_ = 0, fx_ = 0, fy_ = 0; // Camera intrinsics
    cv::Mat *image_ = nullptr;                // reference image

    // private:
    //     static int image_height_;
    //     static int image_width_;
};

#endif // TRACKING_H
