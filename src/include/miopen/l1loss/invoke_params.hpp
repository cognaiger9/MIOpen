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
#pragma once

#include "miopen/miopen.h"
#include <cstddef>
#include <miopen/invoke_params.hpp>
#include <miopen/tensor.hpp>

#include <limits>

namespace miopen {

namespace l1loss {

struct InvokeParams : public miopen::InvokeParams
{
    InvokeParams() = default;

    const TensorDescriptor* iDesc  = nullptr;
    const TensorDescriptor* tDesc  = nullptr;
    const TensorDescriptor* oDesc  = nullptr;
    const TensorDescriptor* diDesc = nullptr;
    const TensorDescriptor* dtDesc = nullptr;
    const TensorDescriptor* doDesc = nullptr;

    ConstData_t i      = nullptr;
    ConstData_t t      = nullptr;
    Data_t o           = nullptr;
    Data_t i_grad      = nullptr;
    Data_t t_grad      = nullptr;
    ConstData_t o_grad = nullptr;

    miopenL1LossReduction_t reduction = MIOPEN_L1LOSS_MEAN_REDUCTION;
    Data_t workspace                  = nullptr;
    std::size_t workspace_size        = 0;

    std::size_t GetWorkspaceSize() const { return workspace_size; }
    Data_t GetWorkspace() const { return workspace; }
};

} // namespace l1loss

} // namespace miopen