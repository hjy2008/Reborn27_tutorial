#include <zenoh.hxx>
#include <opencv2/opencv.hpp>

using namespace std;

int main() {
    auto session = zenoh::Session::open(zenoh::Config::create_default());
    auto publisher = session.declare_publisher(zenoh::KeyExpr("l2/topic"));
    cv::VideoCapture capture(0);
    cv::Mat frame;

    while (capture.read(frame)) {
        vector<uchar> buffer;
        cv::imencode(".jpg", frame, buffer);
        publisher.put(zenoh::Bytes(buffer));
    }
    return 0;
}
