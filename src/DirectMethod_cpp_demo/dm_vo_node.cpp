#include "dm_vo_node.hpp"
// #include "dm_algo_g2o.h"

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp> // imshow

using std::placeholders::_1;

DirectMethodDemoNode::DirectMethodDemoNode(DirectMethod::System *pVO)
    : Node("DirectMethod_cpp_demo_node"),
      m_VO(pVO),
      _cnt(0)
{
    rgb_sub = std::make_shared<message_filters::Subscriber<ImageMsg>>(
        std::shared_ptr<rclcpp::Node>(this), "/camera/infra1/image_rect_raw");
    depth_sub = std::make_shared<message_filters::Subscriber<ImageMsg>>(
        std::shared_ptr<rclcpp::Node>(this), "/camera/depth/image_rect_raw");

    syncApproximate = std::make_shared<message_filters::Synchronizer<approximate_sync_policy>>(
        approximate_sync_policy(10), *rgb_sub, *depth_sub);

    syncApproximate->registerCallback(&DirectMethodDemoNode::GrabIRD, this);
}

DirectMethodDemoNode::~DirectMethodDemoNode()
{
    // TODO
}

void DirectMethodDemoNode::GrabIRD(
    const ImageMsg::SharedPtr msgRGB,
    const ImageMsg::SharedPtr msgD)
{
    // Copy the ros rgb image message to cv::Mat.
    try
    {
        cv_ptrRGB = cv_bridge::toCvShare(msgRGB);
    }
    catch (cv_bridge::Exception &e)
    {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }

    // Copy the ros depth image message to cv::Mat.
    try
    {
        cv_ptrD = cv_bridge::toCvShare(msgD);
    }
    catch (cv_bridge::Exception &e)
    {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }

    RCLCPP_INFO(this->get_logger(), "# received frame %d", ++_cnt);

    // cv::imshow("rgb", cv_ptrRGB->image);
    // cv::imshow("dep", cv_ptrD->image);
    // cv::waitKey(1);

    // // cv::Mat Tcw = m_SLAM->TrackRGBD(cv_ptrRGB->image, cv_ptrD->image, msgRGB->header.stamp.sec);
    cv::Mat Tcw = m_VO->TrackDirectMethod_RGBD(cv_ptrRGB->image, cv_ptrD->image, msgRGB->header.stamp.sec);
    std::cout << Tcw << endl;
}
