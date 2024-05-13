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
#ifndef GUARD_CPU_L1LOSS_HPP
#define GUARD_CPU_L1LOSS_HPP

#include "ford.hpp"
#include "tensor_holder.hpp"
#include <cstddef>
#include <miopen/tensor_view_5d.hpp>

template <class T>
void cpu_l1loss_reduced_forward(tensor<T> input,
                                tensor<T> target,
                                tensor<T>& ref_output,
                                tensor<T>& ref_workspace,
                                float divisor)
{
    auto inputSize = input.desc.GetElementSize();

    /* Phase 1: Calc loss for each element (unreduced) */
    par_ford(inputSize)([&](size_t i) {
        ref_workspace[i] = abs(input[i] - target[i]);
    });

    /* Phase 2: Reduce */
    T res = 0.0f;
    par_ford(inputSize)([&](size_t o) {
        res += ref_workspace[o];
    });
    
    ref_output[0] = res / divisor;
}

template <class T>
void cpu_l1loss_reduced_backward(tensor<T> input,
                                tensor<T> target,
                                tensor<T> dO,
                                tensor<T>& ref_dI,
                                tensor<T>& ref_dT,
                                float divisor)
{
    // Treat contiguous tensors as non-contiguous tensors (for consistency)
    auto I_tv  = get_inner_expanded_tv(input.desc);
    auto T_tv  = get_inner_expanded_tv(target.desc);
    auto dI_tv = get_inner_expanded_tv(ref_dI.desc);
    auto dT_tv = get_inner_expanded_tv(ref_dT.desc);

    auto size = input.desc.GetElementSize();

    par_ford(size)([&](size_t i) {
        uint64_t n[5];
        GET_NCDHW(n[0], n[1], n[2], n[3], n[4], i, I_tv);

        size_t Iidx = TV5D_IDX(I_tv, n[0], n[1], n[2], n[3], n[4]);
        size_t Tidx = TV5D_IDX(T_tv, n[0], n[1], n[2], n[3], n[4]);

        T sub  = input[Iidx] - target[Tidx];
        T grad = static_cast<T>(0.0f);

        if(fabs(sub) < beta)
            grad = sub / beta * dO[0] / divisor;
        else
            grad = (sub >= 0 ? 1.0f : -1.0f) * dO[0] / divisor;

        ref_dI[TV5D_IDX(dI_tv, n[0], n[1], n[2], n[3], n[4])] = grad;
        ref_dT[TV5D_IDX(dT_tv, n[0], n[1], n[2], n[3], n[4])] = -grad;
    });
}

#endif // GUARD_CPU_L1LOSS_HPP
