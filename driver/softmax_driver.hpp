/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
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
#ifndef GUARD_MIOPEN_SOFTMAX_DRIVER_HPP
#define GUARD_MIOPEN_SOFTMAX_DRIVER_HPP

#include "InputFlags.hpp"
#include "driver.hpp"
#include "mloSoftmaxHost.hpp"
#include "random.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"
#include "util_driver.hpp"

#include <../test/verify.hpp>

#include <miopen/miopen.h>
#include <miopen/tensor.hpp>

#include <cfloat>
#include <cstdlib>
#include <memory>
#include <vector>

template <typename Tgpu, typename Tref>
class SoftmaxDriver : public Driver
{
public:
    SoftmaxDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&inputTensor);
        miopenCreateTensorDescriptor(&outputTensor);

        miopenCreateTensorDescriptor(&dInputTensor);
        miopenCreateTensorDescriptor(&dOutputTensor);

        data_type = miopen_type<Tgpu>{};
    }

    std::vector<int> ComputeStrides(std::vector<int> input);
    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;
    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU();

    int RunBackwardGPU() override;
    int RunBackwardCPU();

    Tref GetTolerance();

    int VerifyBackward() override;
    int VerifyForward() override;
    ~SoftmaxDriver() override
    {

        miopenDestroyTensorDescriptor(outputTensor);
        miopenDestroyTensorDescriptor(inputTensor);

        miopenDestroyTensorDescriptor(dOutputTensor);
        miopenDestroyTensorDescriptor(dInputTensor);
    }

private:
    InputFlags inflags;

    int forw;
    bool isContiguous;

    miopenTensorDescriptor_t inputTensor;
    miopenTensorDescriptor_t outputTensor;

    std::unique_ptr<GPUMem> in_dev;
    std::unique_ptr<GPUMem> out_dev;

    std::vector<Tgpu> in;
    std::vector<Tgpu> out;
    std::vector<Tref> outhost;

    miopenTensorDescriptor_t dInputTensor;
    miopenTensorDescriptor_t dOutputTensor;

    std::unique_ptr<GPUMem> din_dev;
    std::unique_ptr<GPUMem> dout_dev;

    std::vector<Tgpu> din;
    std::vector<Tgpu> dout;
    std::vector<Tref> dinhost;

    float alpha;
    float beta;
    miopenSoftmaxAlgorithm_t algo;
    miopenSoftmaxMode_t mode;
};

template <typename Tgpu, typename Tref>
int SoftmaxDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }

    forw = inflags.GetValueInt("forw");
    if(forw != 0 && forw != 1 && forw != 2)
    {
        MIOPEN_THROW("Invalid Forward|Backward Mode");
    }

    alpha        = static_cast<float>(inflags.GetValueDouble("alpha"));
    beta         = static_cast<float>(inflags.GetValueDouble("beta"));
    algo         = miopenSoftmaxAlgorithm_t(inflags.GetValueInt("algorithm"));
    mode         = miopenSoftmaxMode_t(inflags.GetValueInt("mode"));
    isContiguous = inflags.GetValueInt("is_contiguous") == 0 ? false : true;
    return miopenStatusSuccess;
}

// Equivalent to: tensor.tranpose(0, -1).contiguous().tranpose(0, -1) incase contiguous = False
template <typename Tgpu, typename Tref>
std::vector<int> SoftmaxDriver<Tgpu, Tref>::ComputeStrides(std::vector<int> inputDim)
{
    if(!isContiguous)
        std::swap(inputDim.front(), inputDim.back());
    std::vector<int> strides(inputDim.size());
    strides.back() = 1;
    for(int i = inputDim.size() - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * inputDim[i + 1];
    if(!isContiguous)
        std::swap(strides.front(), strides.back());
    return strides;
}

template <typename Tgpu, typename Tref>
int SoftmaxDriver<Tgpu, Tref>::GetandSetData()
{
    std::vector<int> in_len = inflags.GetValueTensor("input_dims").lengths;
    auto in_stride          = ComputeStrides(in_len);
    SetTensorNd(inputTensor, in_len, in_stride, data_type);
    SetTensorNd(outputTensor, in_len, in_stride, data_type);
    SetTensorNd(dInputTensor, in_len, in_stride, data_type);
    SetTensorNd(dOutputTensor, in_len, in_stride, data_type);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SoftmaxDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw",
                         'F',
                         "0",
                         "Run only Forward Softmax (1) | only Backward Softmax (2) | both Forward "
                         "and Backward (0) (Default=0)",
                         "int");
    inflags.AddTensorFlag("input_dims",
                          'I',
                          "2x32x128x128",
                          "The dimensional lengths of the input tensor (Default=2x32x128x128)");
    inflags.AddInputFlag("alpha", 'A', "1.0", "Softmax shift (Default=1.0)", "float");
    inflags.AddInputFlag("beta", 'B', "0.0", "Softmax scale (Default=0.0)", "float");
    inflags.AddInputFlag("algorithm",
                         'a',
                         "1",
                         "softmax algorithms: fast (0), accurate (1), logsoftmax (2) (Default=1)",
                         "int");
    inflags.AddInputFlag(
        "mode", 'm', "1", "instance mode (0), channel mode (1) (Default=1)", "int");
    inflags.AddInputFlag(
        "is_contiguous", 'C', "1", "Is Tensor Contiguous (1) or not (0) (Default=1)", "int");
    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SoftmaxDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{

    size_t in_sz  = GetTensorSize(inputTensor);
    size_t out_sz = GetTensorSize(outputTensor);

    DEFINE_CONTEXT(ctx);
#if MIOPEN_BACKEND_OPENCL
    clGetCommandQueueInfo(q, CL_QUEUE_CONTEXT, sizeof(cl_context), &ctx, nullptr);
#endif
    in_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    out_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));

    din_dev  = std::unique_ptr<GPUMem>(new GPUMem(ctx, in_sz, sizeof(Tgpu)));
    dout_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));

    in      = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    out     = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
    outhost = std::vector<Tref>(out_sz, static_cast<Tref>(0));

    din     = std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0));
    dout    = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
    dinhost = std::vector<Tref>(in_sz, static_cast<Tref>(0));

    for(int i = 0; i < in_sz; i++)
    {
        in[i] = prng::gen_canonical<Tgpu>();
    }

    const Tgpu Data_scale = static_cast<Tgpu>(0.001);
    for(int i = 0; i < out_sz; i++)
    {
        dout[i] = Data_scale * prng::gen_A_to_B(static_cast<Tgpu>(-0.5), static_cast<Tgpu>(0.5));
    }

    status_t status;
    status = in_dev->ToGPU(q, in.data());
    status |= out_dev->ToGPU(q, out.data());

    status |= din_dev->ToGPU(q, din.data());
    status |= dout_dev->ToGPU(q, dout.data());

    if(status != STATUS_SUCCESS)
        printf("Error copying data to GPU\n");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SoftmaxDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;
    float wall_first_time   = 0.0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenSoftmaxForward_V2(GetHandle(),
                                &alpha,
                                inputTensor,
                                in_dev->GetMem(),
                                &beta,
                                outputTensor,
                                out_dev->GetMem(),
                                algo,
                                mode);

        float time = 0.0;
        miopenGetKernelTime(GetHandle(), &time);
        kernel_total_time += time;
        if(i == 0)
        {
            kernel_first_time = time;
            STOP_TIME
            wall_first_time = t.gettime_ms();
            START_TIME
        }
    }

    if(inflags.GetValueInt("time") == 1)
    {
        STOP_TIME
        int iter           = inflags.GetValueInt("iter");
        auto gpu_time      = kernel_first_time;
        auto wall_time     = wall_first_time;
        auto aux_wall_time = 0.0f;
        if(iter > 1)
        {
            gpu_time      = (kernel_total_time - kernel_first_time) / (iter - 1);
            wall_time     = t.gettime_ms() / (iter - 1);
            aux_wall_time = wall_first_time - wall_time;
        }

        if(WALL_CLOCK)
        {
            printf("Wall-clock Time Forward Softmax Elapsed: %f ms", wall_time);
            if(iter > 1)
                printf(", First Call Overhead: %f ms", aux_wall_time);
            printf("\n");
        }
        printf("GPU Kernel Time Forward Softmax Elapsed: %f ms\n", gpu_time);
    }

    out_dev->FromGPU(GetStream(), out.data());

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SoftmaxDriver<Tgpu, Tref>::RunForwardCPU()
{
    return (0);
}

template <typename Tgpu, typename Tref>
int SoftmaxDriver<Tgpu, Tref>::RunBackwardGPU()
{
    float kernel_total_time = 0.0;
    float kernel_first_time = 0.0;
    float wall_first_time   = 0.0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenSoftmaxBackward_V2(GetHandle(),
                                 &alpha,
                                 outputTensor,
                                 out_dev->GetMem(),
                                 dOutputTensor,
                                 dout_dev->GetMem(),
                                 &beta,
                                 dInputTensor,
                                 din_dev->GetMem(),
                                 algo,
                                 mode);

        float time = 0.0;
        miopenGetKernelTime(GetHandle(), &time);
        kernel_total_time += time;
        if(i == 0)
        {
            kernel_first_time = time;
            STOP_TIME
            wall_first_time = t.gettime_ms();
            START_TIME
        }
    }

    if(inflags.GetValueInt("time") == 1)
    {
        STOP_TIME
        int iter           = inflags.GetValueInt("iter");
        auto gpu_time      = kernel_first_time;
        auto wall_time     = wall_first_time;
        auto aux_wall_time = 0.0f;
        if(iter > 1)
        {
            gpu_time      = (kernel_total_time - kernel_first_time) / (iter - 1);
            wall_time     = t.gettime_ms() / (iter - 1);
            aux_wall_time = wall_first_time - wall_time;
        }

        if(WALL_CLOCK)
        {
            printf("Wall-clock Time Backward Softmax Elapsed: %f ms", wall_time);
            if(iter > 1)
                printf(", First Call Overhead: %f ms", aux_wall_time);
            printf("\n");
        }
        printf("GPU Kernel Time Backward Softmax Elapsed: %f ms\n", gpu_time);
    }

    din_dev->FromGPU(GetStream(), din.data());

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
Tref SoftmaxDriver<Tgpu, Tref>::GetTolerance()
{
    Tref tolerance = std::numeric_limits<Tgpu>::epsilon() * 10;
    return tolerance;
}

template <typename Tgpu, typename Tref>
int SoftmaxDriver<Tgpu, Tref>::VerifyForward()
{
    mloSoftmaxForwardRunHost<Tgpu, Tref>(
        inputTensor, outputTensor, in.data(), outhost.data(), alpha, beta, algo, mode);

    auto error           = miopen::rms_range(outhost, out);
    const Tref tolerance = GetTolerance();
    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Forward Softmax FAILED: " << error << std::endl;
    }
    else
    {
        printf("Forward Softmax Verifies on CPU and GPU (err=%f)\n", error);
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SoftmaxDriver<Tgpu, Tref>::RunBackwardCPU()
{

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int SoftmaxDriver<Tgpu, Tref>::VerifyBackward()
{
    mloSoftmaxBackwardRunHost<Tgpu, Tref>(inputTensor,
                                          outputTensor,
                                          out.data(),
                                          dout.data(),
                                          dinhost.data(),
                                          alpha,
                                          beta,
                                          algo,
                                          mode);

    auto error           = miopen::rms_range(dinhost, din);
    const Tref tolerance = GetTolerance();

    if(!std::isfinite(error) || error > tolerance)
    {
        std::cout << "Backward Softmax FAILED: " << error << std::endl;
    }
    else
    {
        printf("Backward Softmax Verifies on CPU and GPU (err=%f)\n", error);
    }

    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_SOFTMAX_DRIVER_HPP
