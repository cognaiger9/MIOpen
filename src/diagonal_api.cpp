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

#include "miopen/miopen.h"
#include "miopen/tensor.hpp"
#include <miopen/diagonal.hpp>
#include <miopen/errors.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>
#include <miopen/tensor_ops.hpp>

extern "C" miopenStatus_t miopenDiagForward(miopenHandle_t handle,
                                            const miopenTensorDescriptor_t inputDesc,
                                            void* input,
                                            const miopenTensorDescriptor_t outputDesc,
                                            void* output,
                                            int64_t diagonal)
{
    MIOPEN_LOG_FUNCTION(handle, inputDesc, input, outputDesc, output, diagonal);

    return miopen::try_([&] {
        miopen::DiagForward(miopen::deref(handle),
                            miopen::deref(inputDesc),
                            DataCast(input),
                            miopen::deref(outputDesc),
                            DataCast(output),
                            diagonal);
    });
}

extern "C" miopenStatus_t miopenDiagFlatForward(miopenHandle_t handle,
                                                const miopenTensorDescriptor_t inputDesc,
                                                void* input,
                                                const miopenTensorDescriptor_t outputDesc,
                                                void* output,
                                                int64_t offset)
{
    MIOPEN_LOG_FUNCTION(handle, inputDesc, input, outputDesc, output, offset);

    return miopen::try_([&] {
        miopen::DiagFlatForward(miopen::deref(handle),
                                miopen::deref(inputDesc),
                                DataCast(input),
                                miopen::deref(outputDesc),
                                DataCast(output),
                                offset);
    });
}

extern "C" miopenStatus_t miopenDiagEmbedForward(miopenHandle_t handle,
                                                 const miopenTensorDescriptor_t inputDesc,
                                                 void* input,
                                                 const miopenTensorDescriptor_t outputDesc,
                                                 void* output,
                                                 int64_t offset,
                                                 int64_t dim1,
                                                 int64_t dim2)
{
    MIOPEN_LOG_FUNCTION(handle, inputDesc, input, outputDesc, output, offset, dim1, dim2);

    return miopen::try_([&] {
        miopen::DiagEmbedForward(miopen::deref(handle),
                                 miopen::deref(inputDesc),
                                 DataCast(input),
                                 miopen::deref(outputDesc),
                                 DataCast(output),
                                 offset,
                                 dim1,
                                 dim2);
    });
}
