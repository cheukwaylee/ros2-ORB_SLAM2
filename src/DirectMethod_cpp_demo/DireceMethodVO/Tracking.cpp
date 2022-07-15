
#include "Tracking.h"

// #include <opencv2/core/core.hpp>
// #include <opencv2/features2d/features2d.hpp>

// #include "ORBmatcher.h"
// #include "FrameDrawer.h"
// #include "Converter.h"
// #include "Map.h"
// #include "Initializer.h"

// #include "Optimizer.h"
// #include "PnPsolver.h"

// #include <iostream>

// #include <mutex>
// #include <unistd.h> // for usleep


namespace DirectMethod
{

    Tracking::Tracking(System *pSys, const string &strSettingPath) : mpSystem(pSys)
    {
        mPoints = 1800;
        mboarder = 10;
        mT_curr_last = Eigen::Isometry3d::Identity(); // res: current wrt last
        mT_curr_world = Eigen::Isometry3d::Identity();
        mfirst_frame_flag = true;

        // Load camera parameters from settings file
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
        mfx = fSettings["Camera.fx"];
        mfy = fSettings["Camera.fy"];
        mcx = fSettings["Camera.cx"];
        mcy = fSettings["Camera.cy"];

        mK << mfx, 0.f, mcx, 0.f, mfy, mcy, 0.f, 0.f, 1.0f;

        cv::Mat DistCoef(4, 1, CV_32F);
        DistCoef.at<float>(0) = fSettings["Camera.k1"];
        DistCoef.at<float>(1) = fSettings["Camera.k2"];
        DistCoef.at<float>(2) = fSettings["Camera.p1"];
        DistCoef.at<float>(3) = fSettings["Camera.p2"];
        const float k3 = fSettings["Camera.k3"];
        if (k3 != 0)
        {
            DistCoef.resize(5);
            DistCoef.at<float>(4) = k3;
        }
        DistCoef.copyTo(mDistCoef);

        mbf = fSettings["Camera.bf"];

        float fps = fSettings["Camera.fps"];
        if (fps == 0)
            fps = 30;

        // Max/Min Frames to insert keyframes and to check relocalisation
        mMinFrames = 0;
        mMaxFrames = fps;

        cout << endl
             << "Camera Parameters: " << endl;
        cout << "- fx: " << mfx << endl;
        cout << "- fy: " << mfy << endl;
        cout << "- cx: " << mcx << endl;
        cout << "- cy: " << mcy << endl;
        cout << "- k1: " << DistCoef.at<float>(0) << endl;
        cout << "- k2: " << DistCoef.at<float>(1) << endl;
        if (DistCoef.rows == 5)
            cout << "- k3: " << DistCoef.at<float>(4) << endl;
        cout << "- p1: " << DistCoef.at<float>(2) << endl;
        cout << "- p2: " << DistCoef.at<float>(3) << endl;
        cout << "- fps: " << fps << endl;

        int nRGB = fSettings["Camera.RGB"];
        mbRGB = nRGB;

        if (mbRGB)
            cout << "- color order: RGB (ignored if grayscale)" << endl;
        else
            cout << "- color order: BGR (ignored if grayscale)" << endl;

        mDepthMapFactor = fSettings["DepthMapFactor"];
        if (fabs(mDepthMapFactor) < 1e-5)
            mDepthMapFactor = 1;
        else
            mDepthMapFactor = 1.0f / mDepthMapFactor;
    }

    cv::Mat Tracking::GrabImageDirectMethod_RGBD(
        const cv::Mat &imRGB,
        const cv::Mat &imD,
        const double &timestamp)
    {
        mImGray = imRGB;
        mImDepth = imD;
        mtimestamp = timestamp;

        if (mImGray.channels() == 3)
        {
            if (mbRGB)
                cvtColor(mImGray, mImGray, CV_RGB2GRAY);
            else
                cvtColor(mImGray, mImGray, CV_BGR2GRAY);
        }
        else if (mImGray.channels() == 4)
        {
            if (mbRGB)
                cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
            else
                cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
        }

        if ((fabs(mDepthMapFactor - 1.0f) > 1e-5) || mImDepth.type() != CV_32F)
            mImDepth.convertTo(mImDepth, CV_32F, mDepthMapFactor);

        Track();

        // 返回前，交给下一个循环
        mImLastGray = mImGray.clone();
        mImLastDepth = mImDepth.clone();
        return eigen2cv(mT_curr_last).clone();
    }

    void Tracking::Track()
    {
        // chrono::steady_clock::time_point t1_opencv = chrono::steady_clock::now();
        /*void GaussianBlur(
        InputArray src, //输入图片
        OutputArray dst, //输出图片
        Size ksize, // Ksize为高斯滤波器窗口大小 15,15
        double sigmaX, // X方向滤波系数 15
        double sigmaY=0, // Y方向滤波系数 0
        int borderType=BORDER_DEFAULT) // 默认边缘插值方法 4   */
        // cv::GaussianBlur(pic_left, pic_left, cv::Size(argv_2, argv_2), argv_3, argv_4, 4);
        cv::medianBlur(mImGray, mImGray, 10);
        // chrono::steady_clock::time_point t2_opencv = chrono::steady_clock::now();
        // chrono::duration<double> time_used_opencv = chrono::duration_cast<chrono::duration<double>>(t2_opencv - t1_opencv);
        // cout << "opencv time: " << time_used_opencv.count() * 1000 << " ms" << endl;

        if (mImLastGray.empty() || mImLastDepth.empty())
        {
        }
        else
        {
            // pickMeasurement(
            //     last_left, last_depth,
            //     rng,
            //     measurements, // output
            //     false, nPoints, boarder);
            mmeasurements.clear();

            // 1-version
            for (int i = 0; i < mPoints; i++)
            {
                int x = mrng.uniform(mboarder, mImLastGray.cols - mboarder); // don't pick pixels close to boarder
                int y = mrng.uniform(mboarder, mImLastGray.rows - mboarder); // don't pick pixels close to boarder
                // int disparity = disparity_img.at<uchar>(y, x);
                // double depth = fx * baseline / disparity; // you know this is disparity to depth
                float d = float(mImLastDepth.ptr<ushort>(y)[x]);
                float grayscale = float(mImLastGray.ptr<uchar>(y)[x]);
                if (grayscale > IR_WHITE_DOTS_THRESHOLD || d == 0)
                    continue;
                // d /= 5000; // TODO d435i png sclar and depth?
                Eigen::Vector3d p3d = project2Dto3D(x, y, d);
                mmeasurements.push_back(Measurement(p3d, grayscale));
            }

            if (mfirst_frame_flag & !mmeasurements.empty())
            {
                mfirst_frame_measurements = mmeasurements;
                mfirst_frame_flag = false;
            }

            // chrono::steady_clock::time_point t1_DirectM = chrono::steady_clock::now();
            mT_curr_last = Eigen::Isometry3d::Identity(); // res: current wrt last
            // poseEstimationDirect(measurements, &pic_left, K, T_curr_last);
            poseEstimationDirect();
            // chrono::steady_clock::time_point t2_DirectM = chrono::steady_clock::now();
            // chrono::duration<double> time_used_DirectM = chrono::duration_cast<chrono::duration<double>>(t2_DirectM - t1_DirectM);

            mT_curr_world = mT_curr_last * mT_curr_world; // curr wrt last * last wrt world
            // cout << "direct method cost time: " << time_used_DirectM.count() * 1000 << " ms" << endl;
            cout.flags(ios::fixed);
            cout.precision(6);
            cout << "T_curr_world=  @" << mtimestamp << endl
                 << mT_curr_last.matrix() << endl
                 << endl;
            mvRelativePoseResult.push_back(RelativePoseResult(mT_curr_last, mtimestamp));

            // plot the feature points
            cv::Mat img_adjTracking;
            cvtColor(mImGray, img_adjTracking, cv::COLOR_GRAY2RGB);

            cv::Mat img_firstTracking;
            cvtColor(mImGray, img_firstTracking, cv::COLOR_GRAY2RGB);

            // cout << "size of first_frame_measurements " << first_frame_measurements.size() << endl;
            // First_Tracking
            for (Measurement m : mfirst_frame_measurements)
            {
                Eigen::Vector3d p = m.pos_world;
                Eigen::Vector2d pixel_first = project3Dto2D(p(0, 0), p(1, 0), p(2, 0));
                Eigen::Vector3d p2 = mT_curr_world * m.pos_world;
                Eigen::Vector2d pixel_now = project3Dto2D(p2(0, 0), p2(1, 0), p2(2, 0));

                cv::circle(img_firstTracking,
                           cv::Point2f(pixel_first[0], pixel_first[1]), 1,
                           cv::Scalar(0, 250, 0), 1);
                cv::line(img_firstTracking,
                         cv::Point2f(pixel_first[0], pixel_first[1]),
                         cv::Point2f(pixel_now[0], pixel_now[1]),
                         cv::Scalar(0, 250, 0));
            }

            // Adj_Tracking
            for (Measurement m : mmeasurements)
            {
                // if (rand() > RAND_MAX / 5)
                //     continue;
                Eigen::Vector3d p = m.pos_world;
                Eigen::Vector2d pixel_prev = project3Dto2D(p(0, 0), p(1, 0), p(2, 0));
                Eigen::Vector3d p2 = mT_curr_last * m.pos_world;
                Eigen::Vector2d pixel_now = project3Dto2D(p2(0, 0), p2(1, 0), p2(2, 0));
                int width = 480, height = 320; // TODO
                if (pixel_now(0, 0) < 0 ||
                    pixel_now(0, 0) >= width ||
                    pixel_now(1, 0) < 0 ||
                    pixel_now(1, 0) >= height)
                    continue;

                cv::circle(img_adjTracking,
                           cv::Point2f(pixel_now[0], pixel_now[1]), 1,
                           cv::Scalar(0, 250, 0), 1);
                cv::line(img_adjTracking,
                         cv::Point2f(pixel_prev[0], pixel_prev[1]),
                         cv::Point2f(pixel_now[0], pixel_now[1]),
                         cv::Scalar(0, 250, 0));
            }
            // cv::imshow("matching", img_show);
            cv::imshow("Adj_Tracking", img_adjTracking);
            cv::imshow("First_Tracking", img_firstTracking);
            cv::waitKey(1);
        }
    }

    // 直接法估计位姿
    // 输入：测量值（空间点的灰度）， 新的灰度图， 相机内参；
    // 输出：相机位姿
    // 返回：true为成功，false失败
    bool Tracking::poseEstimationDirect(
        // const vector<Measurement> &measurements,
        // cv::Mat *gray,
        // Eigen::Matrix3f &K,
        // Eigen::Isometry3d &Tcw
        //
    )
    {
        // 初始化g2o
        typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 1>> DirectBlock; // 求解的向量是6＊1的

        DirectBlock::LinearSolverType *linearSolver = new g2o::LinearSolverDense<DirectBlock::PoseMatrixType>();

        DirectBlock *solver_ptr = new DirectBlock(std::unique_ptr<DirectBlock::LinearSolverType>(linearSolver));

        g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(std::unique_ptr<DirectBlock>(solver_ptr)); // L-M

        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm(solver);
        optimizer.setVerbose(false); // turn off the optimization info

        g2o::VertexSE3Expmap *pose = new g2o::VertexSE3Expmap();                              // Vertex: optimization variable
        pose->setEstimate(g2o::SE3Quat(mT_curr_last.rotation(), mT_curr_last.translation())); // optimization variable initial guess
        pose->setId(0);
        optimizer.addVertex(pose);

        // 添加边
        // 边：误差项，每一个被采样的像素点都构成一个误差
        int id = 1;
        for (Measurement m : mmeasurements)
        {
            EdgeSE3ProjectDirect *edge = new EdgeSE3ProjectDirect(
                m.pos_world,
                mK(0, 0), mK(1, 1), mK(0, 2), mK(1, 2),
                &mImGray); // Edge: error
            edge->setVertex(0, pose);
            edge->setMeasurement(m.grayscale); // reference frame's grayscale value as Measurement （锚点灰度）
            edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
            edge->setId(id++);
            optimizer.addEdge(edge);
        }
        cout << "edges in graph: " << optimizer.edges().size() << endl;
        optimizer.initializeOptimization();
        optimizer.optimize(30);
        mT_curr_last = pose->estimate();
        return true;
    }

} // namespace DirectMethod

// plus in manifold
void EdgeSE3ProjectDirect::linearizeOplus()
{
    if (level() == 1)
    {
        _jacobianOplusXi = Eigen::Matrix<double, 1, 6>::Zero();
        return;
    }

    VertexSE3Expmap *vtx = static_cast<VertexSE3Expmap *>(_vertices[0]);
    Eigen::Vector3d xyz_trans = vtx->estimate().map(x_world_); // q in book

    double x = xyz_trans[0];
    double y = xyz_trans[1];
    double invz = 1.0 / xyz_trans[2];
    double invz_2 = invz * invz;

    // 3d to 2d
    float u = x * fx_ * invz + cx_;
    float v = y * fy_ * invz + cy_;

    // jacobian from se3 to u,v
    // NOTE that in g2o the Lie algebra is (\omega, \epsilon),
    // where \omega is so(3) and \epsilon the translation
    Eigen::Matrix<double, 2, 6> jacobian_uv_ksai;

    jacobian_uv_ksai(0, 0) = -x * y * invz_2 * fx_;        // 0,3
    jacobian_uv_ksai(0, 1) = (1 + (x * x * invz_2)) * fx_; // 0,4
    jacobian_uv_ksai(0, 2) = -y * invz * fx_;              // 0,5
    jacobian_uv_ksai(0, 3) = invz * fx_;                   // 0,0
    jacobian_uv_ksai(0, 4) = 0;                            // 0,1
    jacobian_uv_ksai(0, 5) = -x * invz_2 * fx_;            // 0,2

    jacobian_uv_ksai(1, 0) = -(1 + y * y * invz_2) * fy_; // 1,3
    jacobian_uv_ksai(1, 1) = x * y * invz_2 * fy_;        // 1,4
    jacobian_uv_ksai(1, 2) = x * invz * fy_;              // 1,5
    jacobian_uv_ksai(1, 3) = 0;                           // 1,0
    jacobian_uv_ksai(1, 4) = invz * fy_;                  // 1,1
    jacobian_uv_ksai(1, 5) = -y * invz_2 * fy_;           // 1,2

    Eigen::Matrix<double, 1, 2> jacobian_pixel_uv;

    jacobian_pixel_uv(0, 0) = (getPixelValue(u + 1, v) - getPixelValue(u - 1, v)) / 2;
    jacobian_pixel_uv(0, 1) = (getPixelValue(u, v + 1) - getPixelValue(u, v - 1)) / 2;
    // jacobian_pixel_uv(0, 0) = (getPixelValue_opt(u + 1, v, u_plus) -
    //                            getPixelValue_opt(u - 1, v, u_minus)) /
    //                           2;
    // jacobian_pixel_uv(0, 1) = (getPixelValue_opt(u, v + 1, v_plus) -
    //                            getPixelValue_opt(u, v - 1, v_minus)) /
    //                           2;

    // 1-by-2 * 2-by-6 = 1-by-6
    _jacobianOplusXi = jacobian_pixel_uv * jacobian_uv_ksai;
}

void EdgeSE3ProjectDirect::computeError()
{
    const VertexSE3Expmap *v = static_cast<const VertexSE3Expmap *>(_vertices[0]);
    // 3d to 2d
    // u = fx * x / z + cx;
    // v = fy * y / z + cy;
    Eigen::Vector3d x_local = v->estimate().map(x_world_);
    float x = x_local[0] * fx_ / x_local[2] + cx_;
    float y = x_local[1] * fy_ / x_local[2] + cy_;

    // check x,y is in the image
    if (x - 4 < 0 || (x + 4) > image_->cols ||
        (y - 4) < 0 || (y + 4) > image_->rows)
    {
        _error(0, 0) = 0.0;
        this->setLevel(1); // 不优化这一项？
    }
    else
    {
        // current frame (to be optimized) - reference frame's grayscale value (as Measurement 锚点灰度)
        float pixelValue = getPixelValue(x, y);
        // if (pixelValue > IR_WHITE_DOTS_THRESHOLD)
        // {
        //     _error(0, 0) = 0.0;
        //     this->setLevel(1); // 不优化这一项？
        // }
        _error(0, 0) = pixelValue - _measurement;
    }
}
