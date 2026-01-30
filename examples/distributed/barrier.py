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
import time
from datetime import datetime

DEVICE = torch.device("spyre")
C10D_BACKEND = "spyreccl"

# Check that the c10d backend was loaded properly
if dist.distributed_c10d.is_backend_available(C10D_BACKEND) is False:
    raise RuntimeError(f"Error: Missing the C10 Backend {C10D_BACKEND}")
if C10D_BACKEND != dist.get_default_backend_for_device("spyre"):
    raise RuntimeError(
        f"Error: Missing a C10 Backend for {'sypre'}! Expected {C10D_BACKEND}"
    )

# Force full load of the 'sypre' device
# Needed because barrier will call the following and issue the error below
# >> Could not run 'aten::empty.memory_format' with arguments from the 'spyre' backend.
torch.manual_seed(0xBEEF)

# Initialize the distributed environment
print("# Initialize Distributed Group ")
# With this process group the barrier will throw the following error
# >> Could not run 'c10d::barrier' with arguments from the 'Autogradspyre' backend.
dist.init_process_group()
# The error does not occur if we attach the c10d backend to the 'cpu' device
# dist.init_process_group(f"cpu:{C10D_BACKEND}")

comm_size = dist.get_world_size()
comm_rank = dist.get_rank()

print(
    f"[{comm_rank} of {comm_size}] Before the barrier: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)

# Add a delay to slow down the barrier. Different at different ranks.
time.sleep(comm_rank % 3)

# Execute the barrier
dist.barrier()

print(
    f"[{comm_rank} of {comm_size}] After the barrier : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)
