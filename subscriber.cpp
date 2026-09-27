#include <zenoh.hxx>
#include <opencv2/opencv.hpp>
using namespace std;


int main() {
    auto session = zenoh::Session::open(zenoh::Config::create_default());
    auto subscriber = session.declare_subscriber("l2/topic",
        [](const zenoh::Sample &sample) {
            vector<uchar> images = sample.get_payload().as_vector();
            cv::Mat image = cv::imdecode(images, cv::IMREAD_COLOR);
            if (!image.empty()) {
                cv::imshow("image", image);
                cv::waitKey(1);
            }
        },
        zenoh::closures::none);
    getchar();
}
