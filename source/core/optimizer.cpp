#include "ml_lib/core/optimizer.h"

// Batch
void BatchOptimizer::step(Matrix& param, const Matrix& grad)
{
    param = param.sub(grad.scale(learning_rate));
}


// Stochastic
void StochasticOptimizer::step(Matrix& param, const Matrix& grad)
{
    param = param.sub(grad.scale(learning_rate));
}


// Mini Batch
void MiniBatchOptimizer::step(Matrix& param, const Matrix& grad)
{
    param = param.sub(grad.scale(learning_rate));
}
