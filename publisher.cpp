#include <zenoh.hxx>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;

int main() {
    auto session = zenoh::Session::open(zenoh::Config::create_default());
    auto publisher = session.declare_publisher(zenoh::KeyExpr("l2/topic"));
    string pic;
    cout << "请输入图片位置：";
    cin >> pic;
    while (true) {
        auto image = cv::imread(pic, cv::IMREAD_COLOR);
        std::vector<uchar> buffer;
        cv::imencode(".jpg", image, buffer);
        publisher.put(zenoh::Bytes(buffer));
    }
    return 0;
}
