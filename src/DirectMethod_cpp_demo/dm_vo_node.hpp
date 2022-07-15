
#ifndef DM_VO_NODE_HPP
#define DM_VO_NODE_HPP

#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"

#include "message_filters/subscriber.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/approximate_time.h"

#include <cv_bridge/cv_bridge.h>

// #include "dm_algo_g2o.h"
// #include <sophus/se3.hpp>
#include "System.h"
#include "Tracking.h"

class DirectMethodDemoNode : public rclcpp::Node
{
public:
    DirectMethodDemoNode(DirectMethod::System *pVO);

    ~DirectMethodDemoNode();

private:
    using ImageMsg = sensor_msgs::msg::Image;
    typedef message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::Image,
        sensor_msgs::msg::Image>
        approximate_sync_policy;

    void GrabIRD(const sensor_msgs::msg::Image::SharedPtr msgRGB,
                 const sensor_msgs::msg::Image::SharedPtr msgD);

    cv_bridge::CvImageConstPtr cv_ptrRGB;
    cv_bridge::CvImageConstPtr cv_ptrD;

    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> rgb_sub;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> depth_sub;

    std::shared_ptr<message_filters::Synchronizer<approximate_sync_policy>> syncApproximate;

    DirectMethod::System *m_VO;
    int _cnt;
};

#endif
