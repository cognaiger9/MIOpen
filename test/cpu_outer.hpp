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

#include <miopen/tensor_view_utils.hpp>
#include "tensor_holder.hpp"
#include "tensor_view.hpp"

template <class T>
void cpu_outer_forward(const tensor<T>& x1, const tensor<T>& x2, tensor<T>& ref_y)
{
    auto y_tv    = miopen::get_inner_expanded_tv<2>(ref_y.desc);
    auto y_numel = ref_y.desc.GetElementSize();

    for(size_t i = 0; i < y_numel; i++)
    {
        tensor_layout_t<2> y_layout(y_tv, i);
        ref_y[y_tv.get_tensor_view_idx(y_layout)] = x1[y_layout.layout[0]] * x2[y_layout.layout[1]];
    }
}
