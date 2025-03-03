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

#include "cpu_outer.hpp"
#include "get_handle.hpp"
#include "random.hpp"
#include "tensor_holder.hpp"
#include "verify.hpp"
#include <gtest/gtest.h>
#include <miopen/miopen.h>
#include <miopen/outer.hpp>

struct OuterTestCase
{
    size_t M;
    size_t N;
    bool isContiguous;

    friend std::ostream& operator<<(std::ostream& os, const OuterTestCase& tc)
    {
        return os << " N:" << tc.N << " M:" << tc.M << " contiguous " << tc.isContiguous;
    }

    OuterTestCase() {}

    OuterTestCase(size_t M_, size_t N_, bool cont_) : M(M_), N(N_), isContiguous(cont_) {}

    std::vector<size_t> ComputeStrides(const std::vector<size_t>& input_dim_) const
    {
        std::vector<size_t> inputDim = input_dim_;
        if(!isContiguous)
            std::swap(inputDim.front(), inputDim.back());
        std::vector<size_t> strides(inputDim.size());
        strides.back() = 1;
        for(int i = inputDim.size() - 2; i >= 0; --i)
            strides[i] = strides[i + 1] * inputDim[i + 1];
        if(!isContiguous)
            std::swap(strides.front(), strides.back());
        return strides;
    }
};

inline std::vector<OuterTestCase> GenFullTestCases()
{
    return {{512, 128, true},
            {512, 256, true},
            {512, 512, true},
            {2048, 128, true},
            {2048, 256, true},
            {2048, 512, true},
            {32768, 32, true},
            {32768, 64, true},
            {32768, 128, true},
            {512, 128, false},
            {512, 256, false},
            {512, 512, false},
            {2048, 128, false},
            {2048, 256, false},
            {2048, 512, false},
            {32768, 32, false},
            {32768, 64, false},
            {32768, 128, false}};
}

template <typename T = float>
struct OuterFwdTest : public ::testing::TestWithParam<OuterTestCase>
{
protected:
    void SetUp() override
    {
        auto&& handle   = get_handle();
        outer_config    = GetParam();
        auto gen_value1 = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 100); };
        auto gen_value2 = [](auto...) { return prng::gen_descreet_uniform_sign<T>(1e-2, 99); };

        auto M = outer_config.M;
        auto N = outer_config.N;

        x1 = tensor<T>{std::vector<size_t>({M})}.generate(gen_value1);
        x2 = tensor<T>{std::vector<size_t>({N})}.generate(gen_value2);

        std::vector<size_t> y_dims{M, N};
        auto y_stride = outer_config.ComputeStrides(y_dims);
        y             = tensor<T>{y_dims, y_stride};
        std::fill(y.begin(), y.end(), std::numeric_limits<T>::quiet_NaN());

        ref_y = tensor<T>{y_dims, y_stride};
        std::fill(ref_y.begin(), ref_y.end(), std::numeric_limits<T>::quiet_NaN());

        x1_dev = handle.Write(x1.data);
        x2_dev = handle.Write(x2.data);
        y_dev  = handle.Write(y.data);
    }

    void RunTest()
    {
        auto&& handle = get_handle();

        cpu_outer_forward<T>(x1, x2, ref_y);
        miopenStatus_t status;

        status = miopen::outer::OuterForward(
            handle, x1.desc, x1_dev.get(), x2.desc, x2_dev.get(), y.desc, y_dev.get());

        EXPECT_EQ(status, miopenStatusSuccess);

        y.data = handle.Read<T>(y_dev, y.data.size());
    }

    void Verify()
    {
        double threshold = std::numeric_limits<T>::epsilon();
        auto error       = miopen::rms_range(ref_y, y);

        EXPECT_EQ(miopen::range_distance(ref_y), miopen::range_distance(y));
        EXPECT_LT(error, threshold * 10);
    }

    OuterTestCase outer_config;

    tensor<T> x1;
    tensor<T> x2;
    tensor<T> y;

    tensor<T> ref_y;

    miopen::Allocator::ManageDataPtr x1_dev;
    miopen::Allocator::ManageDataPtr x2_dev;
    miopen::Allocator::ManageDataPtr y_dev;
};
