# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.distributed as dist

DEVICE = torch.device("spyre")
C10D_BACKEND = "spyreccl"

# Check that the c10d backend was loaded properly
if dist.distributed_c10d.is_backend_available(C10D_BACKEND) is False:
    raise RuntimeError(f"Error: Missing the C10 Backend {C10D_BACKEND}")
if C10D_BACKEND != dist.get_default_backend_for_device("spyre"):
    raise RuntimeError(
        f"Error: Missing a C10 Backend for {'sypre'}! Expected {C10D_BACKEND}"
    )

# Initialize the distributed environment
# Add 'cpu:gloo' since we want to use the backend as well
print("# Initialize Distributed Group ")
dist.init_process_group("cpu:gloo")

comm_size = dist.get_world_size()
comm_rank = dist.get_rank()

# Create input tensor
x = torch.rand(512, 1024, dtype=torch.float16)

# Broadcast on the CPU
print(f"[{comm_rank} of {comm_size}] Broadcast Tensor: CPU")
dist.broadcast(x, 0)

# Send input tensor to Spyre device and broadcast using the collective library
print(f"[{comm_rank} of {comm_size}] Broadcast Tensor: Spyre")
x_device = x.to(DEVICE)
dist.broadcast(x_device, 0)
