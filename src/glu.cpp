/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2024 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <miopen/common.hpp>
#include <miopen/datatype.hpp>
#include <miopen/find_solution.hpp>
#include <miopen/float_equal.hpp>
#include <miopen/kernel_cache.hpp>
#include <miopen/glu/invoke_params.hpp>
#include <miopen/glu/solvers.hpp>
#include <miopen/glu.hpp>
#include <miopen/tensor.hpp>

namespace miopen {

miopenStatus_t GLUForward(Handle& handle,
                          const TensorDescriptor& inputDesc,
                          ConstData_t input,
                          const TensorDescriptor& outputDesc,
                          Data_t output,
                          int64_t dim)
{
    const auto problem = glu::ProblemDescription{inputDesc, outputDesc, dim};

    const auto invoke_params = [&]() {
        auto tmp       = glu::InvokeParams{};
        tmp.type       = InvokeType::Run;
        tmp.inputDesc  = &inputDesc;
        tmp.outputDesc = &outputDesc;
        tmp.input      = input;
        tmp.output     = output;
        tmp.dim        = dim;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"GLUForward"};
    const auto solvers = solver::SolverContainer<solver::glu::GLUForward>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

miopenStatus_t GLUBackward(Handle& handle,
                           const TensorDescriptor& inputDesc,
                           ConstData_t input,
                           const TensorDescriptor& outputGradDesc,
                           ConstData_t outputGrad,
                           const TensorDescriptor& inputGradDesc,
                           Data_t inputGrad,
                           int64_t dim)
{
    const auto problem = glu::ProblemDescription{inputDesc, outputGradDesc, inputGradDesc, dim};

    const auto invoke_params = [&]() {
        auto tmp           = glu::BwdInvokeParams{};
        tmp.type           = InvokeType::Run;
        tmp.inputDesc      = &inputDesc;
        tmp.input          = input;
        tmp.inputGradDesc  = &inputGradDesc;
        tmp.inputGrad      = inputGrad;
        tmp.outputGradDesc = &outputGradDesc;
        tmp.outputGrad     = outputGrad;
        tmp.dim            = dim;
        return tmp;
    }();

    const auto algo    = AlgorithmName{"GLUBackward"};
    const auto solvers = solver::SolverContainer<solver::glu::GLUBackward>{};

    solvers.ExecutePrimitive(handle, problem, algo, invoke_params);

    return miopenStatusSuccess;
}

} // namespace miopen