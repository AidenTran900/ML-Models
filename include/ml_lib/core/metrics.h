#pragma once
#include "../math/matrix.h"

enum class MetricType {
    // Linear
    R2,
    AdjustedR2,
    MSE,
    RMSE,
    MAE,
    // Logistic
    Confusion,
    Accuracy,
    Precision,
    Recall,
    F1
};


class Metric {
    public:
        virtual double compute(const Matrix& y_true, const Matrix& y_pred) const = 0;
        virtual ~Metric() {}
};

class R2Metric : public Metric { 
    public:
        virtual double compute(const Matrix& y_true, const Matrix& y_pred) const override;
};
class AdjustedR2Metric : public Metric {
    private:
        int k;
    public:
        AdjustedR2Metric(int predictors) : k(predictors) {}
        virtual double compute(const Matrix& y_true, const Matrix& y_pred) const override; 
};
class MSEMetric : public Metric {
    public:
        virtual double compute(const Matrix& y_true, const Matrix& y_pred) const override; 
};
class RMSEMetric : public Metric {
    public:
        virtual double compute(const Matrix& y_true, const Matrix& y_pred) const override; 
};
class MAEMetric : public Metric {
    public:
        virtual double compute(const Matrix& y_true, const Matrix& y_pred) const override; 
};


class ConfusionMatrix{
    public:
        Matrix compute(const Matrix& y_true, const Matrix& y_pred) const;
};

class ClassificationMetric {
    public:
        virtual double compute(const Matrix& confusion) const = 0;
        virtual ~ClassificationMetric() {}
};

struct ROCResult {
    std::vector<double> TPR;
    std::vector<double> FPR;
};

class ROCCurve {
    public:
        ROCResult compute(const Matrix& y_true, const Matrix& y_pred, const double resolution = 0.01) const;
};

class AccuracyMetric : public ClassificationMetric {
    public:
        virtual double compute(const Matrix& confusion) const override;
};
class PrecisionMetric : public ClassificationMetric {
    public:
        virtual double compute(const Matrix& confusion) const override;
};
class RecallMetric : public ClassificationMetric {
    public:
        virtual double compute(const Matrix& confusion) const override;
};
class FPRMetric : public ClassificationMetric {
    public:
        virtual double compute(const Matrix& confusion) const override;
};
class F1ScoreMetric : public ClassificationMetric {
    public:
        virtual double compute(const Matrix& confusion) const override;
};

Metric* createMetric(MetricType type);
ClassificationMetric* createClassificationMetric(MetricType type);