
#ifndef SYSTEM_H
#define SYSTEM_H

#include <string>
// #include <thread>
#include <opencv2/core/core.hpp>

#include "Tracking.h"
// #include "FrameDrawer.h"
// #include "MapDrawer.h"
// #include "Map.h"
// #include "LocalMapping.h"
// #include "LoopClosing.h"
// #include "KeyFrameDatabase.h"
// #include "ORBVocabulary.h"
// #include "Viewer.h"

namespace DirectMethod
{
    // class Viewer;
    // class FrameDrawer;
    // class Map;
    class Tracking;
    // class LocalMapping;
    // class LoopClosing;

    class System
    {
    public:
        // Initialize the SLAM system. It launches the Local Mapping, Loop Closing and Viewer threads.
        System(const string &strSettingsFile);

        cv::Mat TrackDirectMethod_RGBD(const cv::Mat &im,
                                       const cv::Mat &depthmap,
                                       const double &timestamp);

        void SaveTrajectoryTUM(const string &filename);

    private:
        // The Tracking thread "lives" in the main execution thread that creates the System object.
        Tracking *mpTracker;
    };

} // namespace DirectMethod

#endif // SYSTEM_H
