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

#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/trace.hpp>
#include <miopen/logger.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor_ops.hpp>

extern "C" miopenStatus_t
miopenGetTraceForwardWorkspaceSize(miopenHandle_t handle,
                                   const miopenTensorDescriptor_t inputDesc,
                                   const miopenTensorDescriptor_t outputDesc,
                                   size_t* sizeInBytes)
{
    MIOPEN_LOG_FUNCTION(handle, inputDesc, sizeInBytes);

    return miopen::try_([&] {
        miopen::deref(sizeInBytes) = miopen::trace::GetTraceForwardWorkspaceSize(
            miopen::deref(handle), miopen::deref(inputDesc), miopen::deref(outputDesc));
    });
}

extern "C" miopenStatus_t miopenTraceForward(miopenHandle_t handle,
                                             void* workspace,
                                             size_t workspaceSizeInBytes,
                                             const miopenTensorDescriptor_t inputDesc,
                                             const void* input,
                                             const miopenTensorDescriptor_t outputDesc,
                                             void* output)
{
    MIOPEN_LOG_FUNCTION(
        handle, workspace, workspaceSizeInBytes, inputDesc, input, outputDesc, output);

    return miopen::try_([&] {
        miopen::trace::TraceForward(miopen::deref(handle),
                                    DataCast(workspace),
                                    workspaceSizeInBytes,
                                    miopen::deref(inputDesc),
                                    DataCast(input),
                                    miopen::deref(outputDesc),
                                    DataCast(output));
    });
}

extern "C" miopenStatus_t miopenTraceBackward(miopenHandle_t handle,
                                              const miopenTensorDescriptor_t outputGradDesc,
                                              const void* outputGrad,
                                              const miopenTensorDescriptor_t inputGradDesc,
                                              void* inputGrad)
{
    MIOPEN_LOG_FUNCTION(handle, outputGradDesc, outputGrad, inputGradDesc, inputGrad);

    return miopen::try_([&] {
        miopen::trace::TraceBackward(miopen::deref(handle),
                                     miopen::deref(outputGradDesc),
                                     DataCast(outputGrad),
                                     miopen::deref(inputGradDesc),
                                     DataCast(inputGrad));
    });
}
