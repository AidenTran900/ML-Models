#include "ml_lib/core/metrics.h"
#include <cmath>

double R2Metric::compute(const Matrix& y_pred, const Matrix& y_true) const
{
    const double rows = y_pred.rows();
    const double cols = y_pred.cols();

    double SSres = 0.0; // Residual sum of squares
    double SStot = 0.0; // Total sum of squares

    double mean = 0.0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mean += y_true(i, j);
        }
    }
    mean /= (rows * cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            double diff1 = y_pred(i, j) - y_true(i, j);
            double diff2 = y_true(i, j) - mean;
            SSres += diff1 * diff1;
            SStot += diff2 * diff2;
        }
    }

    if (SStot < 1e-9) { return 0; }

    return 1 - (SSres/SStot);
}

double AdjustedR2Metric::compute(const Matrix& y_pred, const Matrix& y_true) const
{
    R2Metric r2;
    double R2_val = r2.compute(y_true, y_pred);
    int n = y_true.rows();

    return 1 - (1 - R2_val) * ((n - 1)/(n - k - 1));
}

double MSEMetric::compute(const Matrix& y_pred, const Matrix& y_true) const
{
    double result = 0.0;
    int n = y_pred.rows() * y_pred.cols();
    if (n == 0) {
        return 0.0;
    }

    for (int i = 0; i < y_pred.rows(); i++) {
        for (int j = 0; j < y_pred.cols(); j++) {
            double diff = y_pred(i, j) - y_true(i, j);
            result += diff * diff;
        }
    }
    return result / n;
}

double RMSEMetric::compute(const Matrix& y_pred, const Matrix& y_true) const
{
    double result = 0.0;
    int n = y_pred.rows() * y_pred.cols();
    if (n == 0) {
        return 0.0;
    }

    for (int i = 0; i < y_pred.rows(); i++) {
        for (int j = 0; j < y_pred.cols(); j++) {
            double diff = y_pred(i, j) - y_true(i, j);
            result += diff * diff;
        }
    }

    return sqrt(result / n);
}

double MAEMetric::compute(const Matrix& y_pred, const Matrix& y_true) const
{
    double result = 0.0;
    int n = y_pred.rows() * y_pred.cols();
    if (n == 0) {
        return 0.0;
    }

    for (int i = 0; i < y_pred.rows(); i++) {
        for (int j = 0; j < y_pred.cols(); j++) {
            result += abs(y_pred(i, j) - y_true(i, j));
        }
    }
    return result / n;
}


Matrix ConfusionMatrix::compute(const Matrix &y_true, const Matrix &y_pred) const
{
    Matrix confusion = Matrix(2, 2, 0);
    for (int i = 0; i < y_pred.rows(); i++) {
        for (int j = 0; j < y_pred.cols(); j++) {
            double pred_val = y_pred(i, j);
            double true_val = y_true(i, j);

            if (pred_val == true_val) {
                if (true_val == 1) {
                    confusion(0, 0) += 1; // TP
                } else {
                    confusion(1, 1) += 1; // TN
                }
            } else {
                if (true_val == 1) {
                    confusion(0, 1) += 1; // FN
                } else {
                    confusion(1, 0) += 1; // FP
                }
            }
        }
    }
    return confusion;
}

ROCResult ROCCurve::compute(const Matrix &y_true, const Matrix &y_pred, const double resolution) const {
    ROCResult result;

    ConfusionMatrix confusionMatrix;
    RecallMetric recallMetric;
    FPRMetric fprMetric;

    for (double threshold = 0.0; threshold <= 1.0; threshold += resolution) {
        Matrix y_pred_thresholded = Matrix(y_pred.rows(), y_pred.cols(), 0.0);
        for (int i = 0; i < y_pred.rows(); i++) {
            for (int j = 0; j < y_pred.cols(); j++) {
                y_pred_thresholded(i, j) = (y_pred(i, j) >= threshold) ? 1.0 : 0.0;
            }
        }

        Matrix confusion = confusionMatrix.compute(y_true, y_pred_thresholded);

        double tpr = recallMetric.compute(confusion);
        double fpr = fprMetric.compute(confusion);

        result.TPR.push_back(tpr);
        result.FPR.push_back(fpr);
    }

    return result;
}

double AccuracyMetric::compute(const Matrix& confusion) const
{
    int TP = confusion(0, 0);
    int TN = confusion(1, 1);
    int FP = confusion(1, 0);
    int FN = confusion(0, 1);

    int total = TP + TN + FP + FN;
    if (total == 0) {
        return 0.0;
    }
    return static_cast<double>(TP + TN) / total;
}

double PrecisionMetric::compute(const Matrix& confusion) const
{
    int TP = confusion(0, 0);
    int FP = confusion(1, 0);

    if (TP + FP == 0) {
        return 0.0;
    }
    return static_cast<double>(TP) / (TP + FP);
}

double RecallMetric::compute(const Matrix& confusion) const
{
    int TP = confusion(0, 0);
    int FN = confusion(0, 1);

    if (TP + FN == 0) {
        return 0.0;
    }
    return static_cast<double>(TP) / (TP + FN);
}

double FPRMetric::compute(const Matrix& confusion) const
{
    int TN = confusion(1, 1);
    int FP = confusion(1, 0);

    if (FP + TN == 0) {
        return 0.0;
    }
    return static_cast<double>(FP) / (FP + TN);
}

double F1ScoreMetric::compute(const Matrix& confusion) const
{
    RecallMetric recallMetric;
    PrecisionMetric precisionMetric;

    double recall = recallMetric.RecallMetric::compute(confusion);
    double precision = precisionMetric.PrecisionMetric::compute(confusion);

    if (precision + recall == 0.0) {
        return 0.0;
    }
    return 2.0 * (precision * recall) / (precision + recall);
}


Metric* createMetric(MetricType type)
{
    switch (type) {
        case MetricType::MAE:
            return new MAEMetric();
        case MetricType::MSE:
            return new MSEMetric();
        case MetricType::RMSE:
            return new RMSEMetric();
        case MetricType::R2:
            return new R2Metric();
        case MetricType::AdjustedR2:
            return new AdjustedR2Metric(0);
        default:
            return nullptr;
    }
}

ClassificationMetric* createClassificationMetric(MetricType type)
{
    switch (type) {
        case MetricType::Accuracy:
            return new AccuracyMetric();
        case MetricType::Precision:
            return new PrecisionMetric();
        case MetricType::Recall:
            return new RecallMetric();
        case MetricType::F1:
            return new F1ScoreMetric();
        default:
            return nullptr;
    }
}