#include "dm_algo_g2o.h"

#include <opencv2/core/eigen.hpp> // eigen2cv

// Camera intrinsics
float fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
float depth_scale = 1.0;
// baseline
float baseline = 0.573;

cv::Mat dm_algo::Track(const cv::Mat &currentRGB, const cv::Mat &currentDepth)
{

    if (mLastRGB.empty() || mLastDepth.empty())
    {
        cout << "last info empty" << endl;
        mLastRGB = currentRGB.clone();
        mLastDepth = currentDepth.clone();
        return cv::Mat();
    }

    // cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);
    measurements.clear();
    keypoints.clear();

    detector->detect(mLastRGB, keypoints);
    for (auto kp : keypoints)
    {
        // 去掉邻近边缘处的点
        if (kp.pt.x < 20 || kp.pt.y < 20 ||
            (kp.pt.x + 20) > mLastRGB.cols ||
            (kp.pt.y + 20) > mLastRGB.rows)
            continue;
        float d = mLastDepth.ptr<float>(cvRound(kp.pt.y))[cvRound(kp.pt.x)];
        if (d == 0)
            continue;
        Eigen::Vector3d p3d = project2Dto3D(
            kp.pt.x, kp.pt.y, d,
            fx, fy, cx, cy, 1.0F);
        float grayscale = float(mLastRGB.ptr<uchar>(cvRound(kp.pt.y))[cvRound(kp.pt.x)]);
        measurements.push_back(Measurement(p3d, grayscale));
    }

    // last frame info finish!
    mLastRGB = currentRGB.clone();
    mLastDepth = currentDepth.clone();

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    Tcw = Eigen::Isometry3d::Identity();
    poseEstimationDirect(measurements, &mLastRGB, mK, Tcw);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "direct method costs time: " << time_used.count() << " seconds." << endl;

    // MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})
    cv::Mat Tcw_res;
    eigen2cv(Tcw.matrix(), Tcw_res);
    return Tcw_res;
}

// 直接法估计位姿
// 输入：测量值（空间点的灰度），新的灰度图，相机内参；
// 输出：相机位姿
// 返回：true为成功，false失败
bool dm_algo::poseEstimationDirect(
    const vector<Measurement> &measurements,
    cv::Mat *gray,
    Eigen::Matrix3f &K,
    Eigen::Isometry3d &Tcw)
{
    // 初始化g2o
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 1>> DirectBlock; // 求解的向量是6＊1的

    DirectBlock::LinearSolverType *linearSolver = new g2o::LinearSolverDense<DirectBlock::PoseMatrixType>();

    // debug
    // DirectBlock *solver_ptr = new DirectBlock(linearSolver);
    DirectBlock *solver_ptr = new DirectBlock(std::unique_ptr<DirectBlock::LinearSolverType>(linearSolver));

    // g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( solver_ptr ); // G-N
    // g2o::OptimizationAlgorithmGaussNewton *solver = new g2o::OptimizationAlgorithmGaussNewton(std::unique_ptr<DirectBlock>(solver_ptr)); // G-N
    // g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr); // L-M
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(std::unique_ptr<DirectBlock>(solver_ptr)); // L-M

    /* debug tips
    //g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(linearSolver);  // line 356
    g2o::BlockSolver_6_3 * solver_ptr = new g2o::BlockSolver_6_3(std::unique_ptr<g2o::BlockSolver_6_3::LinearSolverType> (linearSolver));
    //g2o::OptimizationAlgorithmLevenberg * solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr); // line 357
    g2o::OptimizationAlgorithmLevenberg * solver = new g2o::OptimizationAlgorithmLevenberg(std::unique_ptr<DirectBlock> (solver_ptr));
    */

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    g2o::VertexSE3Expmap *pose = new g2o::VertexSE3Expmap();
    pose->setEstimate(g2o::SE3Quat(Tcw.rotation(), Tcw.translation()));
    pose->setId(0);
    optimizer.addVertex(pose);

    // 添加边
    // 边：误差项，每一个被采样的像素点都构成一个误差
    int id = 1;
    for (Measurement m : measurements)
    {
        EdgeSE3ProjectDirect *edge = new EdgeSE3ProjectDirect(
            m.pos_world,
            K(0, 0), K(1, 1), K(0, 2), K(1, 2),
            gray);
        edge->setVertex(0, pose);
        edge->setMeasurement(m.grayscale);
        edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity());
        edge->setId(id++);
        optimizer.addEdge(edge);
    }
    cout << "edges in graph: " << optimizer.edges().size() << endl;
    optimizer.initializeOptimization();
    optimizer.optimize(30);
    Tcw = pose->estimate();
}
