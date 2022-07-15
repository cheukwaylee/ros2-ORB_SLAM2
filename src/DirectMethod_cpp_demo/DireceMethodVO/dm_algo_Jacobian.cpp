#include "dm_algo_Jacobian.h"

#include <opencv2/core/eigen.hpp> // eigen2cv

// Camera intrinsics
double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
// baseline
double baseline = 0.573;

// cv::Mat mLastRGB;
// cv::Mat mLastDepth;

dm_algo::dm_algo()
{
}

dm_algo::~dm_algo()
{
}

cv::Mat dm_algo::Track(const cv::Mat &currentRGB, const cv::Mat &currentDepth)
{

    if (mLastRGB.empty() || mLastDepth.empty())
    {
        cout << "last info empty" << endl;
        mLastRGB = currentRGB.clone();
        mLastDepth = currentDepth.clone();
        return cv::Mat();
    }
    cv::imshow("mLastRGB", mLastRGB);
    cv::waitKey(1);

    bool need_kf = true;
    if (need_kf)
    {
        mpixels_ref.clear();
        mdepth_ref.clear();

        // let's randomly pick pixels in the first image and generate some 3d points in the first image's frame
        // generate pixels in ref and load depth data
        for (int i = 0; i < mnPoints; i++)
        {
            int x = mrng.uniform(mboarder, mLastRGB.cols - mboarder); // don't pick pixels close to boarder
            int y = mrng.uniform(mboarder, mLastRGB.rows - mboarder); // don't pick pixels close to boarder
            int depth = mLastDepth.at<uchar>(y, x);
            mpixels_ref.push_back(Eigen::Vector2d(x, y));
            mdepth_ref.push_back(depth);
        }
    }

    Sophus::SE3d T_cur_ref;
    // DirectPoseEstimationSingleLayer(left_img, img, pixels_ref, depth_ref, T_cur_ref);
    DirectPoseEstimationSingleLayer(mLastRGB, currentRGB, mpixels_ref, mdepth_ref, T_cur_ref);
    // DirectPoseEstimationMultiLayer(mLastRGB, currentRGB, mpixels_ref, mdepth_ref, T_cur_ref);

    mLastRGB = currentRGB.clone();
    mLastDepth = currentDepth.clone();

    cv::Mat Tcw;
    eigen2cv(T_cur_ref.matrix(), Tcw);
    return Tcw;
}

void DirectPoseEstimationSingleLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    Sophus::SE3d &T21)
{

    const int iterations = 10;
    double cost = 0, lastCost = 0;
    auto t1 = chrono::steady_clock::now();
    JacobianAccumulator jaco_accu(img1, img2, px_ref, depth_ref, T21);

    for (int iter = 0; iter < iterations; iter++)
    {
        jaco_accu.reset();
        // cv::parallel_for_(cv::Range(0, px_ref.size()),
        //                   std::bind(&JacobianAccumulator::accumulate_jacobian,
        //                             &jaco_accu,
        //                             std::placeholders::_1));
        // cout << "before accumulate_jacobian" << endl;
        jaco_accu.accumulate_jacobian();
        Matrix6d H = jaco_accu.hessian();
        Vector6d b = jaco_accu.bias();

        // solve update and put it into estimation
        Vector6d update = H.ldlt().solve(b);

        T21 = Sophus::SE3d::exp(update) * T21;
        cost = jaco_accu.cost_func();

        if (std::isnan(update[0]))
        {
            // sometimes occurred when we have a black or white patch and H is irreversible
            cout << "update is nan" << endl;
            break;
        }
        if (iter > 0 && cost > lastCost)
        {
            cout << "cost increased: " << cost << ", " << lastCost << endl;
            break;
        }
        if (update.norm() < 1e-3)
        {
            // converge
            break;
        }

        lastCost = cost;
        cout << "iteration: " << iter << ", cost: " << cost << endl;
    }

    cout << "T21 = \n"
         << T21.matrix() << endl;
    auto t2 = chrono::steady_clock::now();
    auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "direct method for single layer: " << time_used.count() << endl;

    // // plot the projected pixels here
    // cv::Mat img2_show;
    // cv::cvtColor(img2, img2_show, CV_GRAY2BGR);
    // VecVector2d projection = jaco_accu.projected_points();
    // for (size_t i = 0; i < px_ref.size(); ++i)
    // {
    //     auto p_ref = px_ref[i];
    //     auto p_cur = projection[i];
    //     if (p_cur[0] > 0 && p_cur[1] > 0)
    //     {
    //         cv::circle(img2_show, cv::Point2f(p_cur[0], p_cur[1]), 2, cv::Scalar(0, 250, 0), 2);
    //         cv::line(img2_show, cv::Point2f(p_ref[0], p_ref[1]), cv::Point2f(p_cur[0], p_cur[1]),
    //                  cv::Scalar(0, 250, 0));
    //     }
    // }
    // cv::imshow("current", img2_show);
    // cv::waitKey();
}

void DirectPoseEstimationMultiLayer(
    const cv::Mat &img1,
    const cv::Mat &img2,
    const VecVector2d &px_ref,
    const vector<double> depth_ref,
    Sophus::SE3d &T21)
{

    // parameters
    int pyramids = 4;
    double pyramid_scale = 0.5;
    double scales[] = {1.0, 0.5, 0.25, 0.125};

    // create pyramids
    vector<cv::Mat> pyr1, pyr2; // image pyramids
    for (int i = 0; i < pyramids; i++)
    {
        if (i == 0)
        {
            pyr1.push_back(img1);
            pyr2.push_back(img2);
        }
        else
        {
            cv::Mat img1_pyr, img2_pyr;
            cv::resize(pyr1[i - 1], img1_pyr,
                       cv::Size(pyr1[i - 1].cols * pyramid_scale, pyr1[i - 1].rows * pyramid_scale));
            cv::resize(pyr2[i - 1], img2_pyr,
                       cv::Size(pyr2[i - 1].cols * pyramid_scale, pyr2[i - 1].rows * pyramid_scale));
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);
        }
    }

    double fxG = fx, fyG = fy, cxG = cx, cyG = cy; // backup the old values
    for (int level = pyramids - 1; level >= 0; level--)
    {
        VecVector2d px_ref_pyr; // set the keypoints in this pyramid level
        for (auto &px : px_ref)
        {
            px_ref_pyr.push_back(scales[level] * px);
        }

        // scale fx, fy, cx, cy in different pyramid levels
        fx = fxG * scales[level];
        fy = fyG * scales[level];
        cx = cxG * scales[level];
        cy = cyG * scales[level];
        DirectPoseEstimationSingleLayer(pyr1[level], pyr2[level], px_ref_pyr, depth_ref, T21);
    }
}

// void JacobianAccumulator::accumulate_jacobian(const cv::Range &range)
// {
//     // parameters
//     const int half_patch_size = 1;
//     int cnt_good = 0;
//     Matrix6d hessian = Matrix6d::Zero();
//     Vector6d bias = Vector6d::Zero();
//     double cost_tmp = 0;

//     for (size_t i = range.start; i < range.end; i++)
//     {
//         // compute the projection in the second image
//         Eigen::Vector3d point_ref =
//             depth_ref[i] * Eigen::Vector3d((px_ref[i][0] - cx) / fx, (px_ref[i][1] - cy) / fy, 1);
//         Eigen::Vector3d point_cur = T21 * point_ref;
//         if (point_cur[2] < 0) // depth invalid
//             continue;

//         float u = fx * point_cur[0] / point_cur[2] + cx, v = fy * point_cur[1] / point_cur[2] + cy;
//         if (u < half_patch_size || u > img2.cols - half_patch_size || v < half_patch_size ||
//             v > img2.rows - half_patch_size)
//             continue;

//         projection[i] = Eigen::Vector2d(u, v);
//         double X = point_cur[0], Y = point_cur[1], Z = point_cur[2],
//                Z2 = Z * Z, Z_inv = 1.0 / Z, Z2_inv = Z_inv * Z_inv;
//         cnt_good++;

//         // and compute error and jacobian
//         for (int x = -half_patch_size; x <= half_patch_size; x++)
//             for (int y = -half_patch_size; y <= half_patch_size; y++)
//             {
//                 double error = GetPixelValue(img1, px_ref[i][0] + x, px_ref[i][1] + y) -
//                                GetPixelValue(img2, u + x, v + y);
//                 Matrix26d J_pixel_xi;
//                 Eigen::Vector2d J_img_pixel;

//                 J_pixel_xi(0, 0) = fx * Z_inv;
//                 J_pixel_xi(0, 1) = 0;
//                 J_pixel_xi(0, 2) = -fx * X * Z2_inv;
//                 J_pixel_xi(0, 3) = -fx * X * Y * Z2_inv;
//                 J_pixel_xi(0, 4) = fx + fx * X * X * Z2_inv;
//                 J_pixel_xi(0, 5) = -fx * Y * Z_inv;

//                 J_pixel_xi(1, 0) = 0;
//                 J_pixel_xi(1, 1) = fy * Z_inv;
//                 J_pixel_xi(1, 2) = -fy * Y * Z2_inv;
//                 J_pixel_xi(1, 3) = -fy - fy * Y * Y * Z2_inv;
//                 J_pixel_xi(1, 4) = fy * X * Y * Z2_inv;
//                 J_pixel_xi(1, 5) = fy * X * Z_inv;

//                 J_img_pixel = Eigen::Vector2d(
//                     0.5 * (GetPixelValue(img2, u + 1 + x, v + y) - GetPixelValue(img2, u - 1 + x, v + y)),
//                     0.5 * (GetPixelValue(img2, u + x, v + 1 + y) - GetPixelValue(img2, u + x, v - 1 + y)));

//                 // total jacobian
//                 Vector6d J = -1.0 * (J_img_pixel.transpose() * J_pixel_xi).transpose();

//                 hessian += J * J.transpose();
//                 bias += -error * J;
//                 cost_tmp += error * error;
//             }
//     }

//     if (cnt_good)
//     {
//         // set hessian, bias and cost
//         unique_lock<mutex> lck(hessian_mutex);
//         H += hessian;
//         b += bias;
//         cost += cost_tmp / cnt_good;
//     }
// }

void JacobianAccumulator::accumulate_jacobian()
{
    // parameters
    const int half_patch_size = 1;
    int cnt_good = 0;
    Matrix6d hessian = Matrix6d::Zero();
    Vector6d bias = Vector6d::Zero();
    double cost_tmp = 0;

    int _test_ = 0;
    for (int ii = 0; ii < mpx_size; ii++) // mpx_size
    {
        cout << "enter zhuman " << _test_++ << endl;
        // compute the projection in the second image
        Eigen::Vector3d point_ref =
            depth_ref[ii] * Eigen::Vector3d((px_ref[ii][0] - cx) / fx, (px_ref[ii][1] - cy) / fy, 1);
        Eigen::Vector3d point_cur = T21 * point_ref;
        if (point_cur[2] < 0) // depth invalid
            continue;

        float u = fx * point_cur[0] / point_cur[2] + cx;
        float v = fy * point_cur[1] / point_cur[2] + cy;
        if (u < half_patch_size || u > img2.cols - half_patch_size ||
            v < half_patch_size || v > img2.rows - half_patch_size)
            continue;

        projection[ii] = Eigen::Vector2d(u, v);
        double X = point_cur[0], Y = point_cur[1], Z = point_cur[2],
               Z2 = Z * Z, Z_inv = 1.0 / Z, Z2_inv = Z_inv * Z_inv;
        cnt_good++;

        // and compute error and jacobian
        for (int x = -half_patch_size; x <= half_patch_size; x++)
            for (int y = -half_patch_size; y <= half_patch_size; y++)
            {
                double error = GetPixelValue(img1, px_ref[ii][0] + x, px_ref[ii][1] + y) -
                               GetPixelValue(img2, u + x, v + y);
                Matrix26d J_pixel_xi;
                Eigen::Vector2d J_img_pixel;

                J_pixel_xi(0, 0) = fx * Z_inv;
                J_pixel_xi(0, 1) = 0;
                J_pixel_xi(0, 2) = -fx * X * Z2_inv;
                J_pixel_xi(0, 3) = -fx * X * Y * Z2_inv;
                J_pixel_xi(0, 4) = fx + fx * X * X * Z2_inv;
                J_pixel_xi(0, 5) = -fx * Y * Z_inv;

                J_pixel_xi(1, 0) = 0;
                J_pixel_xi(1, 1) = fy * Z_inv;
                J_pixel_xi(1, 2) = -fy * Y * Z2_inv;
                J_pixel_xi(1, 3) = -fy - fy * Y * Y * Z2_inv;
                J_pixel_xi(1, 4) = fy * X * Y * Z2_inv;
                J_pixel_xi(1, 5) = fy * X * Z_inv;

                J_img_pixel = Eigen::Vector2d(
                    0.5 * (GetPixelValue(img2, u + 1 + x, v + y) - GetPixelValue(img2, u - 1 + x, v + y)),
                    0.5 * (GetPixelValue(img2, u + x, v + 1 + y) - GetPixelValue(img2, u + x, v - 1 + y)));

                // total jacobian
                Vector6d J = -1.0 * (J_img_pixel.transpose() * J_pixel_xi).transpose();

                hessian += J * J.transpose();
                bias += -error * J;
                cost_tmp += error * error;
            }
    }

    cnt_good++;
    cout << cnt_good << endl;
    if (cnt_good)
    {
        // cout << "xxxxx" << endl;
        // set hessian, bias and cost
        // unique_lock<mutex> lck(hessian_mutex);

        // H += hessian;
        // b += bias;
        // cost += cost_tmp / cnt_good;

        mH = hessian;
        mb = bias;
        mcost = cost_tmp / cnt_good;
    }

    // Matrix6d _temp_1 = hessian;
    // Vector6d _temp_2 = bias;
    // double _temp_3 = cost_tmp / cnt_good;

    // mH = _temp_1;
}