// mat.hpp —— 极简稠密矩阵（行主序），零第三方依赖。
// 作业六只需要 8x8 以内的规模，因此这里不做任何性能优化，只求正确、可读。
#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace kalman {

class Mat {
public:
    Mat() : rows_(0), cols_(0) {}

    Mat(std::size_t rows, std::size_t cols, double value = 0.0)
        : rows_(rows), cols_(cols), data_(rows * cols, value) {}

    static Mat identity(std::size_t n) {
        Mat m(n, n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            m(i, i) = 1.0;
        }
        return m;
    }

    std::size_t rows() const { return rows_; }
    std::size_t cols() const { return cols_; }
    bool empty() const { return data_.empty(); }

    double& operator()(std::size_t r, std::size_t c) { return data_[r * cols_ + c]; }
    double operator()(std::size_t r, std::size_t c) const { return data_[r * cols_ + c]; }

    Mat operator+(const Mat& o) const {
        check_same_shape(o, "+");
        Mat r(rows_, cols_);
        for (std::size_t i = 0; i < data_.size(); ++i) {
            r.data_[i] = data_[i] + o.data_[i];
        }
        return r;
    }

    Mat operator-(const Mat& o) const {
        check_same_shape(o, "-");
        Mat r(rows_, cols_);
        for (std::size_t i = 0; i < data_.size(); ++i) {
            r.data_[i] = data_[i] - o.data_[i];
        }
        return r;
    }

    Mat operator*(double s) const {
        Mat r(rows_, cols_);
        for (std::size_t i = 0; i < data_.size(); ++i) {
            r.data_[i] = data_[i] * s;
        }
        return r;
    }

    Mat operator*(const Mat& o) const {
        if (cols_ != o.rows_) {
            throw std::runtime_error("Mat::operator* 维度不匹配: " + shape() + " * " + o.shape());
        }
        Mat r(rows_, o.cols_, 0.0);
        for (std::size_t i = 0; i < rows_; ++i) {
            for (std::size_t k = 0; k < cols_; ++k) {
                const double a = (*this)(i, k);
                if (a == 0.0) {
                    continue;
                }
                for (std::size_t j = 0; j < o.cols_; ++j) {
                    r(i, j) += a * o(k, j);
                }
            }
        }
        return r;
    }

    Mat transpose() const {
        Mat r(cols_, rows_);
        for (std::size_t i = 0; i < rows_; ++i) {
            for (std::size_t j = 0; j < cols_; ++j) {
                r(j, i) = (*this)(i, j);
            }
        }
        return r;
    }

    // 仅用于对称正定矩阵的求逆（卡尔曼滤波里的 S 一定是对称正定的）。
    // 用带部分主元的 Gauss-Jordan，并对非正定情形给出明确报错，而不是静默返回垃圾。
    Mat inverse_spd() const {
        if (rows_ != cols_) {
            throw std::runtime_error("Mat::inverse_spd 需要方阵，实际 " + shape());
        }
        const std::size_t n = rows_;
        std::vector<double> a(data_);
        std::vector<double> inv(n * n, 0.0);
        for (std::size_t i = 0; i < n; ++i) {
            inv[i * n + i] = 1.0;
        }

        for (std::size_t col = 0; col < n; ++col) {
            std::size_t pivot = col;
            double best = std::abs(a[col * n + col]);
            for (std::size_t r = col + 1; r < n; ++r) {
                const double v = std::abs(a[r * n + col]);
                if (v > best) {
                    best = v;
                    pivot = r;
                }
            }
            if (best < 1e-14) {
                throw std::runtime_error("Mat::inverse_spd 矩阵奇异或非正定 (pivot ≈ 0)");
            }
            if (pivot != col) {
                for (std::size_t j = 0; j < n; ++j) {
                    std::swap(a[col * n + j], a[pivot * n + j]);
                    std::swap(inv[col * n + j], inv[pivot * n + j]);
                }
            }
            const double d = a[col * n + col];
            for (std::size_t j = 0; j < n; ++j) {
                a[col * n + j] /= d;
                inv[col * n + j] /= d;
            }
            for (std::size_t r = 0; r < n; ++r) {
                if (r == col) {
                    continue;
                }
                const double f = a[r * n + col];
                if (f == 0.0) {
                    continue;
                }
                for (std::size_t j = 0; j < n; ++j) {
                    a[r * n + j] -= f * a[col * n + j];
                    inv[r * n + j] -= f * inv[col * n + j];
                }
            }
        }
        return Mat(n, n, 0.0).assign_from(inv);
    }

    // 对称化：数值误差会让 P 逐渐失去对称性，每步之后强制对称避免发散
    Mat symmetrized() const {
        if (rows_ != cols_) {
            return *this;
        }
        Mat r(rows_, cols_);
        for (std::size_t i = 0; i < rows_; ++i) {
            for (std::size_t j = i; j < cols_; ++j) {
                const double v = 0.5 * ((*this)(i, j) + (*this)(j, i));
                r(i, j) = v;
                r(j, i) = v;
            }
        }
        return r;
    }

private:
    std::string shape() const {
        return "(" + std::to_string(rows_) + "x" + std::to_string(cols_) + ")";
    }

    void check_same_shape(const Mat& o, const char* op) const {
        if (rows_ != o.rows_ || cols_ != o.cols_) {
            throw std::runtime_error(std::string("Mat::operator") + op + " 维度不匹配: " + shape() +
                                     " vs " + o.shape());
        }
    }

    Mat& assign_from(const std::vector<double>& v) {
        data_ = v;
        return *this;
    }

    std::size_t rows_;
    std::size_t cols_;
    std::vector<double> data_;
};

}  // namespace kalman
