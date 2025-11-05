#pragma once
#include "../math/matrix.h"

class Regularizer {
    protected:
        double lambda;
    public:
        virtual double compute(const Matrix& weights) const = 0;
        virtual Matrix gradient(const Matrix& weights) const = 0;

        Regularizer(double l) : lambda(l) {}

        virtual ~Regularizer() {}
};

class L1Regularizer : public Regularizer {
    public:
        L1Regularizer(double l) : Regularizer(l) {}
        double compute(const Matrix& weights) const override {
            double result = 0.0;
            for (int i = 0; i < weights.rows(); i++) {
                for (int j = 0; j < weights.cols(); j++) {
                    result += lambda * std::abs(weights(i, j));
                }
            }
            return result;
        }
        
        Matrix gradient(const Matrix& weights) const override {
            return weights.sign().scale(lambda);
        }
};

class L2Regularizer : public Regularizer {
    public:
        L2Regularizer(double l) : Regularizer(l) {}
        double compute(const Matrix& weights) const override {
            double result = 0.0;
            double half_lambda = (lambda/2);
            for (int i = 0; i < weights.rows(); i++) {
                for (int j = 0; j < weights.cols(); j++) {
                    double weight = weights(i, j);
                    result += half_lambda * weight * weight;
                }
            }
            return result;
        }
        
        Matrix gradient(const Matrix& weights) const override {
            // Derivative of w^2 = 2w
            return weights.scale(lambda);
        }
};