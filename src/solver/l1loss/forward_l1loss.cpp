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

#include "miopen/kernel_info.hpp"
#include "miopen/l1loss/problem_description.hpp"
#include "miopen/miopen.h"
#include "miopen/mlo_internal.hpp"
#include <cstddef>
#include <cstdint>
#include <miopen/datatype.hpp>
#include <miopen/kernel_build_params.hpp>
#include <miopen/l1loss/invoke_params.hpp>
#include <miopen/l1loss/solvers.hpp>
#include <miopen/l1loss.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/tensor_view_5d.hpp>

#define LOCAL_SIZE_FWD 256
#define LOCAL_SIZE_REDUCE_FWD 1024

namespace miopen {

namespace solver {

namespace l1loss {

const auto make_hip_kernel = [](std::vector<size_t> localsize,
                                std::vector<size_t> gridsize,
                                std::string kernel_file,
                                std::string kernel_name,
                                KernelBuildParameters build_params) {
    while(localsize.size() < 3)
        localsize.push_back(1);
    while(gridsize.size() < 3)
        gridsize.push_back(1);
    for(int i = 0; i < localsize.size(); ++i)
        gridsize[i] = AlignUp(gridsize[i], localsize[i]);
    return KernelInfo{
        build_params.GenerateFor(kbp::HIP{}), localsize, gridsize, kernel_file, kernel_name};
};

bool L1LossForward5d::IsApplicable(const ExecutionContext& /*context*/,
                                   const miopen::l1loss::L1LossFwdProblemDescription& problem) const
{
    if(!problem.IsSameType())
        return false;
    if(!problem.IsRightLength())
        return false;
    if(!problem.IsRightStride())
        return false;
    if(!problem.IsSameStride())
        return false;
    if(problem.GetReduction() == MIOPEN_L1LOSS_NONE_REDUCTION)
        return false;
    return true;
}

ConvSolution
L1LossForward5d::GetSolution(const ExecutionContext& /*context*/,
                             const miopen::l1loss::L1LossFwdProblemDescription& problem) const
{
    auto result = ConvSolution{miopenStatusSuccess};

    auto dtype        = problem.GetODesc().GetType();
    auto input_dtype  = miopen::GetDataType(problem.GetIDesc().GetType());
    auto output_dtype = miopen::GetDataType(problem.GetODesc().GetType());
    auto size         = problem.GetIDesc().GetElementSize();

    auto build_params =
        KernelBuildParameters{{"MIOPEN_USE_FP16", static_cast<int>(dtype == miopenHalf)},
                              {"MIOPEN_USE_FP32", static_cast<int>(dtype == miopenFloat)},
                              {"MIOPEN_USE_FP64", static_cast<int>(dtype == miopenDouble)},
                              {"MIOPEN_USE_BFP16", static_cast<int>(dtype == miopenBFloat16)},
                              {"INPUT_TYPE", input_dtype == "bfloat16" ? "ushort" : input_dtype},
                              {"OUTPUT_TYPE", output_dtype == "bfloat16" ? "ushort" : output_dtype},
                              {"REDUCE_SIZE", LOCAL_SIZE_REDUCE_FWD}};

    // Phase 1: Calc loss for each element
    result.construction_params.push_back(make_hip_kernel(
        {LOCAL_SIZE_FWD}, {size}, "MIOpenL1Loss.cpp", "L1LossReducedForward5d", build_params));

    // Phase 2: Reduce
    auto _size = size;
    do
    {
        result.construction_params.push_back(make_hip_kernel(
            {LOCAL_SIZE_REDUCE_FWD}, {_size}, "MIOpenL1Loss.cpp", "LossSum", build_params));
        _size = AlignUp(_size, LOCAL_SIZE_REDUCE_FWD) / LOCAL_SIZE_REDUCE_FWD;
    } while(_size > 1);

    result.invoker_factory = [](const std::vector<Kernel>& kernels) {
        return [=](const Handle& handle_, const AnyInvokeParams& raw_params) {
            decltype(auto) params = raw_params.CastTo<miopen::l1loss::InvokeParams>();
            auto elapsed          = 0.f;

            // Phase 1: Calc loss for each element
            {
                decltype(auto) kernel = handle_.Run(kernels.front());
                auto I_tv             = get_inner_expanded_tv(deref(params.iDesc));
                auto T_tv             = get_inner_expanded_tv(deref(params.tDesc));
                auto size             = params.iDesc->GetElementSize();
                size_t divisor = (params.reduction == MIOPEN_L1LOSS_SUM_REDUCTION) ? 1 : size;

                kernel(params.i, params.t, params.workspace, divisor, I_tv, T_tv);
            }
            if(handle_.IsProfilingEnabled())
                elapsed = handle_.GetKernelTime();

            // Phase 2: Reduce
            auto work_a = params.workspace;
            auto work_b = static_cast<Data_t>(static_cast<char*>(params.workspace) +
                                              deref(params.iDesc).GetElementSize() *
                                                  get_data_size(deref(params.oDesc).GetType()));
            auto size   = deref(params.iDesc).GetElementSize();
            for(int i = 1; i < kernels.size(); ++i)
            {
                decltype(auto) kernel = handle_.Run(kernels[i]);
                if(i + 1 != kernels.size())
                {
                    kernel(work_a, work_b, size);
                    std::swap(work_a, work_b);
                }
                else
                {
                    kernel(work_a, params.o, size);
                }
                size = AlignUp(size, LOCAL_SIZE_REDUCE_FWD) / LOCAL_SIZE_REDUCE_FWD;
                if(handle_.IsProfilingEnabled())
                    elapsed += handle_.GetKernelTime();
            }
            if(handle_.IsProfilingEnabled())
            {
                handle_.ResetKernelTime();
                handle_.AccumKernelTime(elapsed);
            };
        };
    };

    return result;
}

std::size_t
L1LossForward5d::GetWorkspaceSize(const ExecutionContext& /*context*/,
                                  const miopen::l1loss::L1LossFwdProblemDescription& problem) const
{
    if(problem.GetReduction() == MIOPEN_L1LOSS_NONE_REDUCTION)
    {
        return 0;
    }

    return (problem.GetIDesc().GetElementSize() +
            AlignUp(problem.GetIDesc().GetElementSize(), LOCAL_SIZE_REDUCE_FWD) /
                LOCAL_SIZE_REDUCE_FWD) *
           get_data_size(problem.GetODesc().GetType());
}

} // namespace l1loss

} // namespace solver

} // namespace miopen