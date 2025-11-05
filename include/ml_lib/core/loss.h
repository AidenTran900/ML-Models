#pragma once
#include "../math/matrix.h"

class LossFunction {
    public:
        virtual double compute(const Matrix& y_pred, const Matrix& y_true) const = 0;
        virtual Matrix gradient(const Matrix& y_pred, const Matrix& y_true) const = 0;
        virtual ~LossFunction() {}
};

class L1Loss : public LossFunction {
    public:
        double compute(const Matrix& y_pred, const Matrix& y_true) const override;
        Matrix gradient(const Matrix& y_pred, const Matrix& y_true) const override;
};

class MAELoss : public LossFunction {
    public:
        double compute(const Matrix& y_pred, const Matrix& y_true) const override;
        Matrix gradient(const Matrix& y_pred, const Matrix& y_true) const override;
};

class L2Loss : public LossFunction {
    public:
        double compute(const Matrix& y_pred, const Matrix& y_true) const override;
        Matrix gradient(const Matrix& y_pred, const Matrix& y_true) const override;
};

class MSELoss : public LossFunction {
    public:
        double compute(const Matrix& y_pred, const Matrix& y_true) const override;
        Matrix gradient(const Matrix& y_pred, const Matrix& y_true) const override;
};

class RMSELoss : public LossFunction {
    public:
        double compute(const Matrix& y_pred, const Matrix& y_true) const override;
        Matrix gradient(const Matrix& y_pred, const Matrix& y_true) const override;
};

