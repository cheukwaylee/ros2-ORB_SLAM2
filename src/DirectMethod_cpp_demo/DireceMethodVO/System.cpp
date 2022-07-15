#include "System.h"
// #include "Converter.h"
// #include <thread>
#include <iomanip>

#include <unistd.h> // for usleep

namespace DirectMethod
{

    System::System(const string &strSettingsFile)
    {
        // Check settings file
        cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
        if (!fsSettings.isOpened())
        {
            cerr << "Failed to open settings file at: " << strSettingsFile << endl;
            exit(-1);
        }

        // Initialize the Tracking thread
        //(it will live in the main thread of execution, the one that called this constructor)
        mpTracker = new Tracking(this, strSettingsFile);
    }

    cv::Mat System::TrackDirectMethod_RGBD(const cv::Mat &im,
                                           const cv::Mat &depthmap,
                                           const double &timestamp)
    {
        cv::Mat Tcw = mpTracker->GrabImageDirectMethod_RGBD(im, depthmap, timestamp);
        return Tcw;
    }

    void System::SaveTrajectoryTUM(const string &filename)
    {
        // cout << endl
        //      << "Saving camera trajectory to " << filename << " ..." << endl;

        // vector<KeyFrame *> vpKFs = mpMap->GetAllKeyFrames();
        // sort(vpKFs.begin(), vpKFs.end(), KeyFrame::lId);

        // // Transform all keyframes so that the first keyframe is at the origin.
        // // After a loop closure the first keyframe might not be at the origin.
        // cv::Mat Two = vpKFs[0]->GetPoseInverse();

        // ofstream f;
        // f.open(filename.c_str());
        // f << fixed;

        // // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
        // // We need to get first the keyframe pose and then concatenate the relative transformation.
        // // Frames not localized (tracking failure) are not saved.

        // // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
        // // which is true when tracking failed (lbL).
        // list<ORB_SLAM2::KeyFrame *>::iterator lRit = mpTracker->mlpReferences.begin();
        // list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
        // list<bool>::iterator lbL = mpTracker->mlbLost.begin();
        // for (list<cv::Mat>::iterator lit = mpTracker->mlRelativeFramePoses.begin(),
        //                              lend = mpTracker->mlRelativeFramePoses.end();
        //      lit != lend; lit++, lRit++, lT++, lbL++)
        // {
        //     if (*lbL)
        //         continue;

        //     KeyFrame *pKF = *lRit;

        //     cv::Mat Trw = cv::Mat::eye(4, 4, CV_32F);

        //     // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
        //     while (pKF->isBad())
        //     {
        //         Trw = Trw * pKF->mTcp;
        //         pKF = pKF->GetParent();
        //     }

        //     Trw = Trw * pKF->GetPose() * Two;

        //     cv::Mat Tcw = (*lit) * Trw;
        //     cv::Mat Rwc = Tcw.rowRange(0, 3).colRange(0, 3).t();
        //     cv::Mat twc = -Rwc * Tcw.rowRange(0, 3).col(3);

        //     vector<float> q = Converter::toQuaternion(Rwc);

        //     f << setprecision(6) << *lT << " " << setprecision(9) << twc.at<float>(0) << " " << twc.at<float>(1) << " " << twc.at<float>(2) << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
        // }
        // f.close();
        // cout << endl
        //      << "trajectory saved!" << endl;
    }

} // namespace DirectMethod
