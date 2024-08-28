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
#ifndef MIOPEN_DONT_USE_HIP_RUNTIME_HEADERS
#include "hip_atomic.hpp"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#endif

#include "float_types.h"

template <typename TIO, typename TINDEX>
__device__ void GatherV2BackwardKernel(const TIO* outputGrad,
                                       const TINDEX* indices,
                                       TIO* paramGrad,
                                       long gather_dim_size,
                                       long out_size,
                                       long indices_size,
                                       long slice_size,
                                       bool is_axis_zero,
                                       ulong param_grad_off)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= out_size)
        return;

    long batch_i   = 0;
    long indices_i = 0;
    long slice_i   = 0;
    if(is_axis_zero)
    {
        indices_i = gid / slice_size;
        slice_i   = gid - indices_i * slice_size;
    }
    else
    {
        long batch_indices_i = gid / slice_size;
        batch_i              = batch_indices_i / indices_size;
        indices_i            = batch_indices_i - batch_i * indices_size;
        slice_i              = gid - batch_indices_i * slice_size;
    }

    // TINDEX is int ?
    size_t gather_i = GET_VAL(indices, indices_i);

    if(gather_i < gather_dim_size)
    {
        long param_i = (batch_i * gather_dim_size + gather_i) * slice_size + slice_i;
        atomic_add_g_safe(paramGrad + param_i + param_grad_off, GET_VAL(outputGrad, gid));
    }
}

extern "C" __global__ void GatherV2Backward(const IN_OUT_TYPE* outputGrad,
                                            const INDEX_TYPE* indices,
                                            IN_OUT_TYPE* paramGrad,
                                            long gather_dim_size,
                                            long out_size,
                                            long indices_size,
                                            long slice_size,
                                            bool is_axis_zero,
                                            ulong param_grad_off)
{
    Diag2dForwardKernel<IN_OUT_TYPE, INDEX_TYPE>(outputGrad,
                                                 indices,
                                                 paramGrad,
                                                 gather_dim_size,
                                                 out_size,
                                                 indices_size,
                                                 slice_size,
                                                 is_axis_zero,
                                                 param_grad_off);
}

template <typename TIO, typename TINDEX>
__device__ void BatchedGatherV2BackwardKernel(const TIO* outputGrad,
                                              const TINDEX* indices,
                                              TIO* paramGrad,
                                              long outer_size,
                                              long gather_dim_size,
                                              long indices_size,
                                              long slice_size,
                                              long out_size,
                                              bool is_axis_zero,
                                              bool is_batch_dim_zero,
                                              ulong param_grad_off)
{
    size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if(gid >= out_size)
        return;

    long batch_i   = 0;
    long outer_i   = 0;
    long indices_i = 0;
    long slice_i   = 0;

    const long slices_count = gid / slice_size;
    if(is_batch_dim_zero)
    {
        if(is_axis_zero)
        {
            indices_i = slices_count;
        }
        else
        {
            outer_i   = slices_count / indices_size;
            indices_i = slices_count - outer_i * indices_size;
        }
    }
    else
    {
        const long entries_count = slices_count / indices_size;
        if(is_axis_zero)
        {
            batch_i = entries_count;
        }
        else
        {
            batch_i = entries_count / outer_size;
            outer_i = entries_count - batch_i * outer_size;
        }
        indices_i = slices_count - entries_count * slice_size;
    }
    slice_i = gid - slices_count * slice_size;

    size_t gather_i = GET_VAL(indices, batch_i * indices_size + indices_i);

    if(gather_i < gather_dim_size)
    {
        long param_i = ((batch_i * outer_size + outer_i) * gather_dim_size) * slice_size + slice_i;
        atomic_add_g_safe(paramGrad + param_i + param_grad_off, GET_VAl(outputGrad, gid));
    }
}

extern "C" __global__ void BatchedGatherV2Backward(const IN_OUT_TYPE* outputGrad,
                                                   const INDEX_TYPE* indices,
                                                   IN_OUT_TYPE* paramGrad,
                                                   long outer_size,
                                                   long gather_dim_size,
                                                   long indices_size,
                                                   long slice_size,
                                                   long out_size,
                                                   bool is_axis_zero,
                                                   bool is_batch_dim_zero,
                                                   ulong param_grad_off)
{
    BatchedGatherV2BackwardKernel<IN_OUT_TYPE, INDEX_TYPE>(outputGrad,
                                                           indices,
                                                           paramGrad,
                                                           outer_size,
                                                           gather_dim_size,
                                                           indices_size,
                                                           slice_size,
                                                           out_size,
                                                           is_axis_zero,
                                                           is_batch_dim_zero,
                                                           param_grad_off);
}
