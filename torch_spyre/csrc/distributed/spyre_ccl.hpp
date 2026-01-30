/*
 * Copyright 2026 The Torch-Spyre Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <pybind11/chrono.h>
#include <torch/python.h>

#include <torch/csrc/distributed/c10d/Backend.hpp>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/csrc/distributed/c10d/Work.hpp>

namespace c10d {

/***********************************************
 * Wrapper torch.distributed Backend for the Sypre Collective Library
 ***********************************************/
class SpyreCCLBackend : public c10d::Backend {
 public:
  SpyreCCLBackend(const c10::intrusive_ptr<::c10d::Store>& store, int rank,
                  int size);

  /*
   * Informative
   */
  const std::string getBackendName() const override {
    return std::string("SpyreCCL");
  }

  /*
   * Allgather
   */
  c10::intrusive_ptr<Work> allgather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  c10::intrusive_ptr<Work> _allgather_base(
      at::Tensor& outputBuffer, at::Tensor& inputBuffer,
      const AllgatherOptions& opts = AllgatherOptions()) override;

  /*
   * Allreduce
   */
  c10::intrusive_ptr<Work> allreduce(
      std::vector<at::Tensor>& tensors,
      const AllreduceOptions& opts = AllreduceOptions()) override;

  c10::intrusive_ptr<Work> allreduce_coalesced(
      std::vector<at::Tensor>& tensors,
      const AllreduceCoalescedOptions& opts =
          AllreduceCoalescedOptions()) override;

  /*
   * Alltoall
   */
  c10::intrusive_ptr<Work> alltoall(
      std::vector<at::Tensor>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  c10::intrusive_ptr<Work> alltoall_base(
      at::Tensor& outputTensor, at::Tensor& inputTensor,
      std::vector<int64_t>& outputSplitSizes,
      std::vector<int64_t>& inputSplitSizes,
      const AllToAllOptions& opts = AllToAllOptions()) override;

  /*
   * Barrier
   */
  c10::intrusive_ptr<Work> barrier(
      const BarrierOptions& opts = BarrierOptions()) override;

  /*
   * Broadcast
   */
  c10::intrusive_ptr<Work> broadcast(
      std::vector<at::Tensor>& data,
      const BroadcastOptions& opts = BroadcastOptions()) override;

  /*
   * Gather
   */
  c10::intrusive_ptr<Work> gather(
      std::vector<std::vector<at::Tensor>>& outputTensors,
      std::vector<at::Tensor>& inputTensors,
      const GatherOptions& opts = GatherOptions()) override;

  /*
   * Reduce
   */
  c10::intrusive_ptr<Work> reduce(
      std::vector<at::Tensor>& tensors,
      const ReduceOptions& opts = ReduceOptions()) override;

  /*
   * Reduce-Scatter
   */
  c10::intrusive_ptr<Work> reduce_scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ReduceScatterOptions& opts = ReduceScatterOptions()) override;

  /*
   * Scatter
   */
  c10::intrusive_ptr<Work> scatter(
      std::vector<at::Tensor>& outputTensors,
      std::vector<std::vector<at::Tensor>>& inputTensors,
      const ScatterOptions& opts = ScatterOptions()) override;

  /*
   * Point-to-Point
   */
  c10::intrusive_ptr<Work> send(std::vector<at::Tensor>& tensors, int dstRank,
                                int tag) override;

  c10::intrusive_ptr<Work> recv(std::vector<at::Tensor>& tensors, int srcRank,
                                int tag) override;

  c10::intrusive_ptr<Work> recvAnysource(std::vector<at::Tensor>& tensors,
                                         int tag) override;

  /*
   * Shutdown
   */
  void abort() {};
  void shutdown() {};

  /*
   * Backend registration
   */
  static c10::intrusive_ptr<Backend> createSpyreCCLBackend(
      const c10::intrusive_ptr<::c10d::Store>& store, int rank, int size,
      const std::chrono::duration<float>& timeout);

  static void SpyreCCLBackendConstructor() __attribute__((constructor)) {
    py::object module = py::module::import("torch.distributed");
    py::object register_backend =
        module.attr("Backend").attr("register_backend");

    // Must be a vector - even if one element
    // https://github.com/pytorch/pytorch/issues/153822
    std::vector<std::string> supported_devices;
    supported_devices.push_back("spyre");

    register_backend("spyreccl", py::cpp_function(createSpyreCCLBackend), false,
                     supported_devices);
  }
};

/***********************************************
 * Wrapper backend for the Sypre Collective Library - Work
 ***********************************************/
class SpyreCCLWork : public Work {
  friend class SpyreCCLBackend;

 public:
  SpyreCCLWork(OpType opType);
  bool isCompleted() override;
  bool isSuccess() const override;
  bool wait(std::chrono::milliseconds timeout = kUnsetTimeout) override;
  virtual c10::intrusive_ptr<c10::ivalue::Future> getFuture() override;

 private:
  c10::intrusive_ptr<c10::ivalue::Future> future_;
};

}  // namespace c10d
