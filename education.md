# torch-spyre Education Guide

## Chapter 1: The Big Picture

### What problem does this repo solve?

When you write a PyTorch model (say, a language model or an image
classifier), it's just Python code that does math on tensors (multi-dimensional
arrays of numbers). By default, that math runs on your CPU or an NVIDIA GPU.

IBM makes an AI chip called **Spyre**. It's a specialized accelerator
designed for AI inference (running a trained model to get predictions, as
opposed to training it). This repo — torch-spyre — is the software that
teaches PyTorch how to run your model on a Spyre chip instead of a CPU or
GPU.

The end goal: a user writes normal PyTorch code, swaps the device to
`"spyre"`, and everything just works.

```python
import torch
import torch_spyre  # this one import makes "spyre" available as a device

# Normal PyTorch code — nothing Spyre-specific after the device string
x = torch.randn(2, 3, dtype=torch.float16, device="spyre")
y = x + x
```

### How does PyTorch support new devices?

PyTorch has a mechanism called **PrivateUse1** — a reserved slot for
exactly one custom device backend. torch-spyre claims that slot and renames
it to `"spyre"`. After that, `torch.device("spyre")` is a real PyTorch
device, with all the expected behavior: you can move tensors to it, run
operations on it, and use `torch.compile` with it.

This is what "out-of-tree backend" means: the Spyre support lives in its
own repository, not inside PyTorch's source code. You install it alongside
PyTorch like a plugin.

### Two ways to run: Eager vs. Compiled

PyTorch has two execution modes, and torch-spyre supports both:

**Eager mode** — operations execute one at a time, immediately, as Python
encounters them. This is the default PyTorch behavior. Each call like
`x + y` dispatches to a Spyre implementation of that operation, runs it,
and returns the result before moving to the next line.

**Compiled mode** (`torch.compile`) — PyTorch first traces your code to
understand the full sequence of operations, then optimizes and compiles
them as a batch before executing. This is where the "AI compiler" part
comes in, and it's where most of the complexity in this repo lives.

```python
@torch.compile
def my_model(x):
    return torch.relu(x @ weight + bias)

# The first call traces, compiles, and then executes.
# Subsequent calls reuse the compiled version.
result = my_model(input_tensor)
```

### The Spyre hardware, briefly

> **Dive deeper:**
> [Spyre Accelerator](docs/source/architecture/spyre_accelerator.md) for
> the full hardware overview, and
> [Dataflow Architecture](docs/source/architecture/dataflow_architecture.md)
> for the dataflow accelerator model.

You don't need to understand chip design, but a few facts help:

- **32 cores** per card, each capable of AI math (matrix operations).
- **Default precision: float16** — Spyre is optimized for half-precision
  floating point. This means 16-bit numbers, not the 32-bit `float32`
  that CPUs default to. This is normal for inference — less precision is
  fine for predictions and it's much faster.
- **Sticks** — Spyre organizes memory in 128-byte aligned chunks called
  "sticks." At float16 (2 bytes per number), one stick holds exactly 64
  elements. This is the fundamental memory unit — tensors must be laid out
  to align with stick boundaries. Think of it like how a hard drive reads
  in blocks, not individual bytes.
- **>300 TOPS at 75W** — it does a lot of math per watt, which is the
  whole point of a specialized accelerator vs. a general-purpose CPU.

### What's in this repo (the 30-second tour)

| Directory | What it does |
|---|---|
| `torch_spyre/` | The Python package. Entry point, device registration, eager ops. |
| `torch_spyre/_inductor/` | The compiler backend — the largest and most complex part. Translates PyTorch's intermediate representation into Spyre instructions. |
| `torch_spyre/csrc/` | C++ code compiled into the `torch_spyre._C` extension module. Provides the hardware runtime: device memory allocator, tensor storage with `SpyreTensorLayout`, kernel execution, and DMA transfers. This is the bridge between Python and the actual Spyre silicon. |
| `torch_spyre/ops/` | Eager-mode operation implementations and CPU fallbacks. |
| `torch_spyre/device/` | PyTorch device interface (how PyTorch discovers and talks to Spyre). |
| `tests/` | Test suites for eager ops, compiled ops, and building blocks. |
| `docs/` | Architecture docs, user guides, and reference material. |

### How it all connects (the life of a tensor operation)

Here's what happens when you do `z = x + y` on a Spyre device:

**In eager mode:**
1. PyTorch's **dispatcher** — a routing system that looks at what
   operation you called and what device the tensor lives on — sees
   this targets a `"spyre"` device.
2. It finds the Spyre implementation registered for `add`.
3. That implementation actually calls `torch.compile` under the hood,
   compiling and running the operation through the full Inductor
   pipeline (see compiled mode below). The result is cached, so
   subsequent calls with the same shapes skip compilation. Spyre's
   hardware operates on compiled programs — there's no way to "just
   run" a single uncompiled op on the chip, so eager mode wraps
   compiled mode transparently.

**In compiled mode** (wrapped in `@torch.compile`):
1. **Dynamo** (PyTorch's tracer) captures the Python code into an FX Graph
   — a data structure representing the operations and their dependencies.
2. **AOTAutograd** strips away training-specific gradient tracking (we're
   doing inference).
3. **Inductor** (PyTorch's compiler backend) lowers the FX Graph into
   Loop-Level IR — a lower-level representation closer to actual compute.
4. **torch-spyre's backend** takes over:
   - Applies Spyre-specific rewrites and decompositions (e.g., breaking a
     complex op into simpler ops Spyre can handle).
   - Figures out how to tile tensors into sticks and distribute work across
     the 32 cores.
   - Generates a JSON specification (called SuperDSC) describing exactly
     what each core should do.
5. **DeepTools** (IBM's proprietary back-end compiler, not in this repo)
   takes that JSON and produces the actual binary that runs on the chip.

The division of labor: torch-spyre handles "what to compute and how to
divide the work," DeepTools handles "how to execute it efficiently on
silicon."

---

## Chapter 2: The Compilation Pipeline

> **Dive deeper:**
> [Compiler Architecture](docs/source/compiler/architecture.md) for the
> canonical compiler architecture reference, and
> [Inductor Front-End Deep Dive](docs/source/compiler/inductor_frontend.md)
> for a detailed look at passes, lowerings, decompositions, and codegen.

Chapter 1 gave you the 10,000-foot view: user code goes through Dynamo,
Inductor, and the Spyre backend. Now let's zoom in on each stage. By the
end of this chapter you should understand what each component does, what
data it produces, and where torch-spyre plugs in.

### Why compile at all?

In eager mode, every operation is a separate round-trip: Python tells the
device "add these two tensors," the device does it, returns the result,
and Python moves on to the next line. This works but it's slow — there's
overhead per operation, and the device sits idle between calls waiting for
Python to tell it what to do next.

Compilation solves this by looking at the *entire* computation upfront.
Instead of sending one-at-a-time instructions, the compiler can:

- **Batch operations** into a single device program (one round-trip
  instead of many).
- **Optimize** by rewriting operations — for example, fusing `x * 2 + 1`
  into a single fused kernel instead of two separate ones.
- **Plan memory and tiling** — decide how to slice tensors across Spyre's
  32 cores and arrange data in memory for maximum throughput.

### Stage 1: Dynamo — "What operations does this Python code perform?"

**Dynamo** (`torch._dynamo`) is PyTorch's tracing system. Its job is
deceptively simple: watch your Python function execute and record every
tensor operation it performs.

The output is an **FX Graph** — a directed acyclic graph (DAG) where each
node is an operation (like `add`, `matmul`, `relu`) and edges represent
data flow between them.

```python
@torch.compile
def f(x, w, b):
    return torch.relu(x @ w + b)
```

Dynamo traces this and produces an FX Graph roughly equivalent to:

```text
%x    = placeholder
%w    = placeholder
%b    = placeholder
%mm   = call_function: aten.mm(%x, %w)
%add  = call_function: aten.add(%mm, %b)
%relu = call_function: aten.relu(%add)
return (%relu,)
```

A few things to notice:

- The operations are **ATen ops** — PyTorch's low-level C++ operation
  library. `torch.relu` in Python becomes `aten.relu` in the graph.
  There are ~2000 ATen ops covering everything from basic arithmetic to
  convolutions.
- The graph captures **data dependencies**, not Python control flow.
  If your code has an `if` statement that depends on tensor values,
  Dynamo can't trace through it statically — it "graph breaks" and
  creates separate graphs for each segment. This isn't fatal — each
  segment gets compiled independently, and Python runs the glue code
  between them in eager mode. But more graph breaks means less
  optimization opportunity, so model authors try to minimize them.
- Dynamo doesn't know or care about Spyre. It just records ATen ops.

**Where torch-spyre hooks in:** Dynamo needs to know basic things about
the device (like "does it exist?"). torch-spyre registers a
`SpyreInterface` (in `torch_spyre/device/interface.py`) so Dynamo can
recognize `"spyre"` tensors. But Dynamo itself doesn't do anything
Spyre-specific — it produces the same FX Graph regardless of device.

### Stage 2: AOTAutograd — "Strip the training machinery"

**AOTAutograd** (Ahead-Of-Time Autograd) takes the FX Graph from Dynamo
and removes the autograd (automatic differentiation) tracking. Since
Spyre targets inference, not training, there are no gradients to compute.
AOTAutograd produces a clean, forward-only FX Graph of ATen ops.

This stage is mostly transparent to torch-spyre — it's upstream PyTorch
machinery that just runs.

### Stage 3: Inductor — "Lower the graph toward hardware"

**Inductor** (`torch._inductor`) is PyTorch's default compiler backend.
For NVIDIA GPUs, Inductor generates Triton kernels. For Spyre, it routes
to torch-spyre's backend instead.

Inductor does several things in sequence:

#### 3a. Decompositions — "Simplify complex ops into primitives"

Many ATen ops are "complex" — they do multiple things internally. For
example, `layer_norm` combines mean, variance, subtraction, division,
and scaling. Hardware accelerators typically implement a smaller set of
**primitive** operations.

**Decompositions** rewrite complex ops into sequences of simpler ones
that the hardware can handle directly.

torch-spyre registers its own decompositions in
`torch_spyre/_inductor/decompositions.py`. These are Spyre-specific
because different hardware supports different primitive sets. For
example, Spyre might have a native `rms_norm` but need to decompose
`layer_norm` into pieces.

The decorator `@register_spyre_decomposition` adds a decomposition that
activates *only* when compiling for Spyre — it doesn't affect CPU or GPU
compilation.

```python
# Simplified example of what a decomposition looks like:
@register_spyre_decomposition([aten.some_complex_op])
def some_complex_op_decomp(input, ...):
    # Rewrite using simpler ops that Spyre supports natively
    temp = aten.mul(input, scale)
    return aten.add(temp, bias)
```

**How decompositions work mechanically:** A decomposition is *not* a
graph rewrite — it doesn't directly mutate or copy the FX Graph.
Instead, it's a **lookup table entry**. When Inductor encounters
`aten.some_complex_op` during tracing/lowering, it checks the
decomposition table, finds the registered function, and *calls* it. That
function executes using simpler ops, and Inductor traces *those* calls
instead. The complex op never makes it into the final graph — it gets
replaced inline during construction, like macro expansion.

The Spyre decomposition table (`spyre_decompositions`) is a plain dict
mapping op → replacement function. It's temporarily merged into
Inductor's global decomposition table via a context manager
(`enable_spyre_decompositions`) that:

1. Adds Spyre-specific decompositions to the table.
2. *Removes* certain Inductor default decompositions that don't work
   on Spyre.
3. Removes decompositions for "fallback ops" (ops that should stay
   intact so they can fall back to eager CPU execution).
4. On exit, restores the original table — so non-Spyre compilations
   are unaffected.

#### 3b. FX Graph Passes — "Rewrite patterns in the graph"

After decompositions, Inductor runs a series of **graph passes** —
transformations that rewrite the FX Graph to optimize it or reshape it
for the target hardware.

torch-spyre hooks into this via two extension points defined in
`torch_spyre/_inductor/passes.py`:

- **`CustomPrePasses`** — runs early, before Inductor's own passes.
  Currently empty (reserved for future use).
- **`CustomPostPasses`** — runs late, after most Inductor passes. This
  is where torch-spyre does things like:
  - Replacing scalar operations with tensor operations (Spyre operates
    on tensors, not scalars).
  - Relaying out weight matrices for more efficient memory access.
  - Converting `mm` (matrix multiply) patterns into `bmm` (batched
    matrix multiply) when beneficial for Spyre's parallelism.

These passes operate on the FX Graph — they look for specific patterns
of nodes and rewrite them into different patterns.

#### 3c. Lowering — "FX Graph → Loop-Level IR"

**Lowering** is the step where abstract operations (like "add these two
tensors") become concrete loop descriptions (like "for each element at
index i, compute output[i] = a[i] + b[i]").

Inductor's **Loop-Level IR** represents computation as nested loops over
tensor elements. Each node becomes either a `Pointwise` operation (does
something at every element independently) or a `Reduction` (combines
elements, like a sum).

To make this concrete, here's what a matrix multiply (`aten.mm`) looks
like when lowered to Loop-Level IR. Given `C = A @ B` where A is [M, K]
and B is [K, N]:

```text
Reduction(
    ranges       = [M, N],           # output dimensions — one element per (row, col)
    reduction    = [K],              # dimension being summed over
    inner_fn     = (index, r_index) → (A[index[0], r_index[0]],
                                       B[r_index[0], index[1]]),
    reduction_type = "matmul"        # tells the backend this is a matmul, not a generic sum
)
```

Read it like pseudocode for these nested loops:

```text
for i in range(M):          # ← ranges[0]
    for j in range(N):      # ← ranges[1]
        for k in range(K):  # ← reduction_ranges[0]
            C[i,j] += A[i,k] * B[k,j]
```

The IR doesn't literally generate loops — it *describes* the iteration
space and what to compute at each point. The backend then decides how to
actually execute it (tiling across cores, memory layout, etc.).

A simpler example — a pointwise operation like `relu(x)`:

```text
Pointwise(
    ranges   = [M, N],              # same shape as the input
    inner_fn = (index) → relu(x[index[0], index[1]])
)
```

No reduction dimension — just "do this at every element." The actual
code for these is in `torch_spyre/_inductor/lowering.py` (see `lower_mm`
at line ~203 for the matmul example).

torch-spyre registers custom lowerings in
`torch_spyre/_inductor/lowering.py` via `@register_spyre_lowering`. This
tells Inductor "when you see `aten.some_op` targeting a Spyre device,
lower it using this specific implementation."

Some ops don't get lowered at all — they **fall back** to eager mode on
the CPU. If Spyre can't handle an operation natively, torch-spyre
unregisters its lowering so that Inductor knows to run it as a regular
eager op instead.

### Stage 4: Scheduling — "What becomes a kernel?"

After lowering, the graph is a collection of Loop-Level IR nodes.
Inductor organizes these into a **scheduler graph** — a dependency graph
where each `SchedulerNode` wraps an IR node and tracks what buffers it
reads and writes. Dependencies are inferred automatically: if node B
reads a buffer that node A writes, B depends on A. The scheduler graph
is the "execution plan" layer — the FX Graph describes *what ops* to do,
the Loop-Level IR describes *how to compute each op*, and the scheduler
graph describes *what order to run them in and what data flows between
them*. When we say the compiler "traverses the scheduler graph in
topological order," it means walking from inputs to outputs, visiting
each node only after all its dependencies are satisfied.

The **scheduler** decides how to group these nodes into **kernels** —
units of work that will be sent to the device as a single program.

A key scheduler decision is **fusion** — combining multiple operations
into a single kernel. Consider `y = relu(x + 1)`. Without fusion, this
is two kernels: one that reads `x` from memory, adds 1, writes the
result back; and a second that reads it back, applies relu, writes
again. With fusion, it becomes one kernel that reads `x` once, does both
operations, and writes once. That's half the memory traffic.

GPU backends fuse aggressively because GPU memory bandwidth is the main
bottleneck — every unnecessary read/write is expensive. Spyre's
scheduler (`SuperDSCScheduling` in `torch_spyre/_inductor/dsc.py`)
deliberately does **no fusion at all**. Every operation stays as its own
kernel.

This isn't laziness — it's an architectural decision about *where*
optimization happens. Spyre uses a two-compiler design: the front-end
(torch-spyre) describes *what* to compute, and the back-end (DeepTools)
decides *how* to execute it on the dataflow hardware. If the front-end
fused operations together, it would be making hardware-level decisions
without enough information — and it would obscure the individual
operations that DeepTools needs to see in order to map them efficiently
onto Spyre's cores. Keeping operations separate preserves the most
information for the back-end compiler to work with.

#### Scheduler Passes — "Plan the physical layout"

Before kernels are finalized, torch-spyre runs its own scheduler-level
passes (registered in `passes.py:scheduler_passes`):

1. **`propagate_spyre_tensor_layouts`** (stickify.py) — figures out
   how each tensor should be laid out in Spyre's stick-based memory.
   Inserts layout conversion operations where tensors need to be
   reformatted.

2. **`core_division_planning`** (core_division.py) — decides how to
   split the work across Spyre's 32 cores. If you have a 1024-element
   tensor and 32 cores, each core might process 32 elements.

3. **`scratchpad_planning`** (scratchpad.py, optional) — plans usage of
   Spyre's on-chip **scratchpad memory**. Each Spyre core has a small,
   fast SRAM (called LX) physically located next to the compute units,
   in addition to the large DDR main memory. Think of it like CPU L1
   cache vs. RAM — scratchpad is much faster but much smaller. This
   pass decides which tensors (or parts of tensors) to place in
   scratchpad for faster access. Only runs when `LX_PLANNING=1`
   because it's still an evolving optimization.

### Stage 5: Kernel Compilation — "Loop-Level IR → OpSpec"

Each scheduled kernel gets compiled by `SpyreKernel`
(`torch_spyre/_inductor/spyre_kernel.py`). This is where the abstract
loop description becomes a concrete **OpSpec** — a data structure that
describes one operation for the hardware.

#### What is an OpSpec?

An `OpSpec` (defined in `torch_spyre/_inductor/op_spec.py`) is a
self-contained description of a single device operation. Think of it as
an instruction card: "perform this operation, on these tensors, with
these dimensions, divided across cores like this."

```python
@dataclass
class OpSpec:
    op: str                    # e.g. "add", "matmul", "relufwd"
    is_reduction: bool         # True for ops that collapse a dimension (sum, matmul)
    iteration_space: list[int] # the output dimensions, e.g. [64, 128]
    iteration_space_dict: dict # maps dimension symbols → (size, core_split)
    args: Sequence[TensorArg]  # input and output tensors with full layout info
    op_info: dict              # op-specific extras (constants, core division, etc.)
```

Each tensor in the operation gets a `TensorArg` describing everything
the back-end compiler needs to know about it:

```python
@dataclass
class TensorArg:
    is_input: bool             # input or output?
    arg_index: int             # position in the kernel's argument list
    device_dtype: DataFormats  # on-device data format (see note below)
    device_size: list[int]     # shape in device memory (stick-aligned)
    device_coordinates: list   # how iteration space maps to device dimensions
    dtype: torch.dtype         # host-side dtype (e.g. torch.float16)
    it_dim_map: list[int]      # maps each iteration dimension to a tensor dimension
                               #   (-1 = broadcast/reduction for this tensor)
    allocation: Any            # scratchpad memory offset, if applicable
    device_layout: SpyreTensorLayout  # full device memory layout descriptor
```

**A note on `DataFormats`:** You'll see `DataFormats.SEN169_FP16`
throughout the codebase. This is Spyre's internal name for its float16
data format — "SEN169" refers to the specific encoding the hardware
uses. For practical purposes, `SEN169_FP16` ≈ `torch.float16`. The
`DataFormats` enum also determines the stick size (64 elements for
fp16, 32 for fp32) since different precisions pack differently into
the 128-byte stick.

For a concrete example, imagine `C = A + B` where A and B are both
[64, 128] float16 tensors. The resulting OpSpec would look roughly like:

```text
OpSpec(
    op = "add",
    is_reduction = False,
    iteration_space = [64, 128],
    args = [
        TensorArg(is_input=True,  ...),   # A
        TensorArg(is_input=True,  ...),   # B
        TensorArg(is_input=False, ...),   # C (output)
    ],
    op_info = { "n_cores_used": 32 }
)
```

#### How SpyreKernel builds OpSpecs

`SpyreKernel` extends Inductor's `Kernel` base class and intercepts the
operations that Inductor would normally turn into GPU code. The
mechanism is a **handler pattern**:

1. Inductor "replays" the Loop-Level IR through the kernel. For each
   operation, it calls methods like `load`, `store`, and named ops
   (`add`, `mul`, `relu`, etc.) on a handler object.

2. For GPUs, the handler emits Triton code strings. For Spyre, the
   handler (`SpyreKernelOpsHandler`) routes to **`SpyreOpFuncs`** — a
   class that maps PyTorch op names to Spyre op names:

```python
class SpyreOpFuncs:
    """Maps PyTorch ops to Spyre backend op names."""

    @staticmethod
    def add(a, b):
        return PointwiseOp("add", [a, b])

    @staticmethod
    def relu(x):
        return PointwiseOp("relufwd", [x])

    @staticmethod
    def mul(a, b):
        return PointwiseOp("mul", [a, b])

    @staticmethod
    def reciprocal(x):
        return PointwiseOp("reciprocal", [x])

    # ... ~30 more ops
```

3. When the handler hits a `store` (writing a result), SpyreKernel
   assembles the full `OpSpec`: it takes the operation from step 2,
   creates `TensorArg`s for each input and the output (with device
   layouts, dimension mappings, and core division info), and appends
   the `OpSpec` to its list.

If an op isn't in `SpyreOpFuncs`, the handler returns an
`UnimplementedOp` — this will cause an error later, signaling that
the op needs to be added to the backend (via a decomposition, a
lowering, or a new `SpyreOpFuncs` entry).

#### Why this matters

The OpSpec is the **boundary between Inductor and Spyre**. Everything
before this point (Dynamo, decompositions, lowering, scheduling) is
PyTorch machinery adapted for Spyre. Everything after (SuperDSC codegen,
DeepTools) is Spyre-specific. The OpSpec is the handshake format — it
captures exactly what the front-end decided (what op, what shapes, what
layout, how to split across cores) in a form the back-end can consume.

### Stage 6: Code Generation — "OpSpec → SuperDSC JSON + Host Code"

This stage produces two separate outputs that work together: a **JSON
specification** for the device (what to compute) and **Python host code**
(how to orchestrate it). Think of it like a concert: the JSON is the
sheet music for the Spyre cores, and the host code is the conductor.

#### Part A: OpSpec → SDSCSpec → SuperDSC JSON

The codegen pipeline in `torch_spyre/_inductor/codegen/` converts each
`OpSpec` into a JSON document called a **SuperDSC** (Super Data Stream
Configuration). This JSON is the input format that DeepTools (the
back-end compiler) understands.

The conversion happens in two steps:

**Step 1: `parse_op_spec`** (in `superdsc.py`) transforms an `OpSpec`
into an intermediate `SDSCSpec`:

```python
@dataclass
class SDSCSpec:
    opfunc: str                     # Spyre op name, e.g. "add", "relufwd", "matmul"
    execution_unit: str             # "sfp" (scalar/vector unit) or "pt" (matrix unit)
    data_format: DataFormats        # e.g. SEN169_FP16
    num_inputs: int                 # how many input tensors
    iteration_space: dict           # dimension symbols → sizes, e.g. {i: 64, j: 128}
    num_cores: int                  # how many cores run this op
    work_slices: dict               # how iteration space is sliced per core
    core_id_to_work_slice: dict     # which slice each core gets (symbolic expressions)
    padding: dict                   # extra elements needed for stick alignment
    layouts: dict                   # unique tensor memory layouts used
    args: list[SDSCArgs]            # per-tensor details (scales, strides, offsets)
    constants: dict                 # op-specific constants (e.g. epsilon, clip bounds)
    coordinate_masking: dict        # masks for padded regions
```

This is where abstract dimension symbols get renamed to the labels
DeepTools expects (like `i`, `j`, `k`), core-to-work mappings are
computed (which slice of the iteration space goes to which core), and
stick-alignment padding is calculated.

**Step 2: `generate_sdsc`** (in `compute_ops.py`) renders the `SDSCSpec`
into the final JSON structure. The JSON encodes, for each core:

- What operation to perform
- The iteration space dimensions and sizes
- Which slice of the data this core owns
- Memory layout descriptors (dimension ordering, stick sizes, strides)
- Coordinate info for each tensor (how to walk through device memory)
- Constants (epsilon values, clipping bounds, etc.)

For a simple `add` of two [64, 128] tensors on 32 cores, the JSON
would describe: "core 0 handles rows 0–1, core 1 handles rows 2–3, ...,
each core adds corresponding elements from input A and input B, writes
to output C."

#### Part B: Wrapper Codegen — the host-side orchestration

While the JSON tells the *device* what to compute, something needs to
run on the *host CPU* to manage the whole process. That's the job of
`SpyrePythonWrapperCodegen` (`torch_spyre/_inductor/wrapper.py`).

This class extends Inductor's `PythonWrapperCodegen` to generate a
Python file that, when executed, does the following:

1. **Allocates device tensors** — with the correct Spyre memory layout
   (stick-aligned, using `spyre_empty_with_layout`):
   ```python
   # Generated code looks like:
   buf0 = spyre_empty_with_layout(
       (64, 128),                    # host shape
       (128, 1),                     # host stride
       torch.float16,                # dtype
       SpyreTensorLayout(...)        # device memory layout descriptor
   )
   ```

2. **Compiles kernels** — sends each SuperDSC JSON to the back-end
   compiler (via `SpyreAsyncCompile`), which can happen in parallel.

3. **Launches kernels** — calls the compiled binaries in the correct
   order with the right tensor arguments.

4. **Handles buffer reuse** — when an output tensor has the same shape
   as a previous intermediate, reuses the memory via
   `reinterpret_tensor_with_layout` instead of allocating fresh.

5. **Returns results** — collects the final output tensors and returns
   them to the calling Python code.

The generated file is regular Python — you can inspect it by setting
`TORCH_COMPILE_DEBUG=1`, which dumps all Inductor artifacts to a debug
directory.

#### Why two outputs?

This split mirrors how real hardware works. The device is a
special-purpose compute engine — it doesn't know about Python, memory
management, or what order to run things. It only knows how to execute
the program binary it's given. The host CPU handles everything else:
memory allocation, data movement, sequencing. The SuperDSC JSON becomes
the device program; the generated Python becomes the host program. They
work together at runtime.

### Stage 7: Back-End Compilation — "JSON → Binary"

> **Dive deeper:**
> [Back-End Compiler](docs/source/compiler/backend.md) for documentation
> on the DeepTools back-end compiler and the SuperDSC format.

The SuperDSC JSON is handed to **DeepTools**, IBM's proprietary back-end
compiler (not in this repo). DeepTools takes the per-core specifications
and produces an optimized binary program that runs directly on the Spyre
silicon. This is analogous to how NVIDIA's `ptxas` compiles PTX assembly
into GPU machine code.

### Putting it all together

Here's the full pipeline with the key files at each stage:

```text
User Python code
    │
    ▼
[Dynamo] ─── traces Python ──→ FX Graph (ATen ops)
    │                            torch_spyre/device/interface.py
    ▼
[AOTAutograd] ─── strips grad tracking ──→ Forward-only FX Graph
    │
    ▼
[Inductor]
    ├─ Decompositions ─────────→ Simplified FX Graph
    │                            torch_spyre/_inductor/decompositions.py
    ├─ FX Graph Passes ────────→ Optimized FX Graph
    │                            torch_spyre/_inductor/passes.py
    ├─ Lowering ───────────────→ Loop-Level IR
    │                            torch_spyre/_inductor/lowering.py
    ▼
[Spyre Backend]
    ├─ Scheduling ─────────────→ Grouped kernels
    │                            torch_spyre/_inductor/dsc.py
    ├─ Scheduler Passes ───────→ Tiled, core-divided kernels
    │                            torch_spyre/_inductor/stickify.py
    │                            torch_spyre/_inductor/core_division.py
    ├─ Kernel Compilation ─────→ OpSpec
    │                            torch_spyre/_inductor/spyre_kernel.py
    ├─ Code Generation ────────→ SuperDSC JSON + host Python
    │                            torch_spyre/_inductor/codegen/superdsc.py
    │                            torch_spyre/_inductor/wrapper.py
    ▼
[DeepTools] ─── proprietary ──→ Device binary
```

### How torch-spyre "wraps" compile_fx

One detail worth understanding: torch-spyre doesn't replace PyTorch's
compiler — it wraps it. In `torch_spyre/_inductor/__init__.py`, there's
a function `enable_spyre_compile_fx_wrapper` that monkey-patches
PyTorch's `compile_fx` function. The wrapper:

1. Checks if the graph contains any Spyre tensors (by inspecting inputs,
   outputs, and graph nodes for `device="spyre"`).
2. If yes, activates a context manager that injects Spyre decompositions
   and configuration.
3. Calls the original `compile_fx` with the Spyre-aware settings.
4. If no Spyre tensors, calls the original `compile_fx` unchanged.

This means torch-spyre is invisible when you're not using Spyre — it
only activates when Spyre tensors are present.

### Key vocabulary recap

| Term | What it is |
|---|---|
| **ATen** | PyTorch's C++ tensor operation library (~2000 ops) |
| **FX Graph** | A DAG of operations captured by Dynamo |
| **Decomposition** | Rewriting a complex op into simpler ops |
| **Lowering** | Translating abstract ops into concrete loop descriptions |
| **Loop-Level IR** | Inductor's representation: nested loops over tensor elements |
| **OpSpec** | torch-spyre's description of one operation for the hardware |
| **SuperDSC** | JSON format consumed by DeepTools back-end compiler |
| **Kernel** | A unit of computation sent to the device as one program |
| **Graph pass** | A transformation that rewrites an FX Graph |

---

## Chapter 3: Tensor Layouts & Sticks

> **Dive deeper:**
> [Tensors and Layouts](docs/source/user_guide/tensors_and_layouts.md)
> for the full tensor layout specification with diagrams, DMA encoding
> details, and layout compatibility rules.

We've mentioned "sticks," "tiling," and `SpyreTensorLayout` throughout
the previous chapters. This chapter explains what they actually are and
why they matter.

### The problem: CPU memory vs. hardware memory

On a CPU, a 2D tensor like `torch.randn(4, 6)` is just 24 numbers laid
out consecutively in memory — row 0 first, then row 1, and so on. PyTorch
tracks this with two vectors:

- **size** = `[4, 6]` — the shape.
- **stride** = `[6, 1]` — how far to jump in memory when stepping along
  each dimension. Moving one row = jump 6 elements. Moving one column =
  jump 1 element.

To find any element, you do a dot product:
`offset(row, col) = row * 6 + col * 1`.

This is clean and simple. But Spyre's hardware doesn't read memory one
element at a time — it reads in **128-byte aligned, 128-byte chunks**
called **sticks**. At float16 (2 bytes per element), one stick is
exactly **64 elements**. Every memory access fetches a full stick.

This means tensors must be physically reorganized in device memory so
that the data each core needs falls on stick boundaries. You can't just
copy the CPU's flat row-major layout onto the device — it would be
misaligned and inefficient.

### What a stick looks like

Imagine a 1D tensor of 256 float16 elements on the CPU:

```text
CPU memory: [e0, e1, e2, ... e63, e64, e65, ... e127, e128, ... e191, e192, ... e255]
             └──────────────────┘  └───────────────────┘  └──────────────────┘  └───────────────────┘
```

On Spyre, the same data is organized as 4 sticks:

```text
Stick 0: [e0   ... e63 ]    ← 128 bytes, 64 elements
Stick 1: [e64  ... e127]    ← 128 bytes, 64 elements
Stick 2: [e128 ... e191]    ← 128 bytes, 64 elements
Stick 3: [e192 ... e255]    ← 128 bytes, 64 elements
```

Each stick is 128-byte aligned in device DDR memory. The hardware
always reads/writes a whole stick at once.

### Tiling: what happens in 2D and beyond

For a 2D tensor, things get more interesting. Consider a [1024, 256]
float16 tensor. On the CPU it's 1024 rows of 256 elements each, stored
row-major. On Spyre:

- The 256-element rows are broken into **4 sticks** of 64 elements each.
- The device layout adds an extra dimension for the tiling.

The CPU shape `[1024, 256]` becomes a device shape like
`[4, 1024, 64]`:

```text
Device dimension 0 (size 4):    which tile (stick group) along the row
Device dimension 1 (size 1024): which row
Device dimension 2 (size 64):   position within a stick
```

The key insight: **the device tensor has more dimensions than the CPU
tensor**. A 2D CPU tensor becomes a 3D device tensor. A 3D CPU tensor
becomes 4D. This extra dimension comes from splitting one of the
original dimensions into tiles.

### SpyreTensorLayout — the metadata

The `SpyreTensorLayout` class (defined in C++, exposed to Python)
captures how a device tensor maps back to its CPU counterpart. It has
three fields:

```python
SpyreTensorLayout(
    device_size = [4, 1024, 64],    # shape in device memory
    dim_map     = [1, 0, 1],        # maps each device dim → CPU dim
    device_dtype = DataFormats.SEN169_FP16
)
```

**`device_size`** is the shape in device memory — more dimensions than
the CPU shape, with sizes adjusted for stick alignment and tiling.

**`dim_map`** maps each device dimension back to the corresponding CPU
dimension. In the example above:
- Device dim 0 (size 4) → CPU dim 1 (the 256-element columns, tiled)
- Device dim 1 (size 1024) → CPU dim 0 (the rows)
- Device dim 2 (size 64) → CPU dim 1 (also columns — within one tile)

When a CPU dimension appears **twice** in `dim_map`, that's the tiling.
Dim 1 appears at device positions 0 and 2, meaning the 256 columns are
split into 4 groups of 64. To reconstruct the original column index:
`col = device_dim_0 * 64 + device_dim_2`.

A `dim_map` value of **-1** indicates a **synthetic dimension** — one
that doesn't correspond to any CPU dimension. This is used for **sparse
tensors**. Here's why they exist: when a reduction op (like `sum` or
`mean`) collapses a dimension, the output might have just one element
where the input had 64 (a full stick). But Spyre still reads and writes
in sticks — you can't have a "partial stick." So the output gets a
synthetic inner dimension of size 64, where only the first element is
meaningful and the rest are padding. The `-1` in `dim_map` marks this
dimension as "doesn't map to any CPU dimension — it's just here for
stick alignment."

### A real example

```python
import torch
x = torch.rand(5, 100, 150, dtype=torch.float16)
y = x.to("spyre")
print(y.device_tensor_layout())

# Output:
# SpyreTensorLayout(
#     device_size=[100, 3, 5, 64],
#     dim_map=[1, 2, 0, 2],
#     device_dtype=DataFormats.SEN169_FP16
# )
```

What happened:
- CPU shape is `[5, 100, 150]` (3D).
- Device shape is `[100, 3, 5, 64]` (4D — one extra dimension from
  tiling).
- The stick dimension is always last: size 64 (elements per stick at
  fp16).
- CPU dim 2 (size 150) got tiled: 150 doesn't divide evenly by 64, so
  it's **padded to 192** (= 3 * 64), then split into device dims 1
  (size 3) and 3 (size 64). The `dim_map` shows both positions mapping
  to CPU dim 2.
- CPU dim 1 (size 100) became device dim 0.
- CPU dim 0 (size 5) became device dim 2.

The **default layout** algorithm: (a) designate the last CPU dimension
as the stick dimension, (b) tile along the first CPU dimension, (c) pad
the stick dimension to be evenly divisible by the stick size (64 for
fp16).

### Padding

Stick alignment often requires padding. If your tensor's last dimension
is 150 elements but sticks hold 64, you need 192 elements (3 sticks) —
42 elements of padding. The padded elements exist in device memory but
don't correspond to real data. The compiler tracks this and uses
**coordinate masking** in the SuperDSC JSON to ensure operations ignore
padded elements where it matters (e.g., reductions that shouldn't sum
padding values).

### FixedTiledLayout — why the compiler needs both layouts

Inductor — PyTorch's compiler — was designed for CPUs and GPUs, where
a tensor's memory layout is fully described by `size` and `stride`.
Internally, Inductor uses a class called `FixedLayout` for every buffer:
to decide how much memory to allocate, to determine if two buffers can
share memory, and to generate the code that reads and writes them.

But for Spyre, `size` and `stride` aren't enough. A [1024, 256] float16
tensor needs `1024 * 256 * 2 = 512 KB` on the host. On the device,
with tiling and padding, it might need `4 * 1024 * 64 * 2 = 512 KB` as
well in this case — but for a [1024, 150] tensor the host needs `300 KB`
while the device needs `1024 * 3 * 64 * 2 = 384 KB` (padding 150 → 192).
If the compiler only saw the host layout, it would:

- **Allocate the wrong amount of memory** on the device.
- **Generate incorrect buffer reuse** — two tensors that look the same
  size on the host might have completely different device layouts and
  can't share memory.
- **Produce wrong codegen** — the wrapper code needs to emit
  `spyre_empty_with_layout(...)` with the actual device layout, not a
  generic `torch.empty(...)`.

`FixedTiledLayout` (`torch_spyre/_inductor/ir.py`) solves this by
carrying *both* representations:

```python
class FixedTiledLayout(FixedLayout):
    def __init__(self, device, dtype, size, stride, device_layout):
        super().__init__(device, dtype, size, stride)
        self.device_layout: SpyreTensorLayout = device_layout
        self.allocation: dict = {}   # scratchpad memory offset, if used
```

The host-side fields (`size`, `stride`) keep Inductor's existing
machinery working — graph passes, buffer lifetime analysis, and
shape propagation all operate on these without modification. The
`device_layout` is used by Spyre-specific codegen for actual device
memory allocation, DMA generation, and SuperDSC output. It's the
adapter that lets an out-of-tree backend plug into Inductor's memory
planning without modifying Inductor itself.

### The stickification pass

The compiler can't assume all tensors arrive with the right layout.
The **stickification pass** (`torch_spyre/_inductor/stickify.py`,
function `propagate_spyre_tensor_layouts`) runs during scheduling
(Chapter 2, Stage 4) and does the following:

1. **Assigns layouts** — traverses the scheduler graph in topological
   order. For each operation, determines the required `SpyreTensorLayout`
   for inputs and outputs based on what the operation needs. For
   example, a pointwise `add` needs both inputs in the same layout; a
   matmul has specific requirements for how the M, N, K dimensions map
   to sticks.

2. **Propagates constraints** — an operation's output layout often
   determines the required input layout of the next operation. Layouts
   flow forward through the graph.

3. **Inserts restickify ops** — when two adjacent operations disagree
   on layout (e.g., op A outputs in layout X, but op B needs layout Y),
   the pass inserts a **restickify** operation between them. This is a
   data movement op that reorganizes sticks in device memory without
   changing the logical tensor values.

### DMA: moving data between host and device

When a tensor is transferred to the Spyre device (via `.to("spyre")`),
the data must be reorganized from CPU row-major layout to the tiled
stick layout. This is done via **DMA (Direct Memory Access)** transfers
encoded as nested loops.

For a [1024, 256] float16 tensor (4 stick tiles per row):

```text
for tile in range(4):           # 4 tiles per row
    for row in range(1024):     # 1024 rows
        for elem in range(64):  # 64 elements per stick
            device[tile*65536 + row*64 + elem] = host[row*256 + tile*64 + elem]
```

Notice the index arithmetic is different on each side — the host reads
sequentially across a row, but the device groups elements by tile first.
These DMA specifications (called **DCIs** — Data Copy Instructions) are
generated automatically from the `SpyreTensorLayout`.

### Why this all matters

Tensor layout is not an afterthought — it's one of the most
performance-critical aspects of Spyre compilation. Getting the layout
wrong means:

- **Wasted memory** from excessive padding.
- **Unnecessary restickify ops** that burn time reorganizing data.
- **Misaligned accesses** that the hardware can't execute at full
  bandwidth.

The stickification pass, core division planning, and layout propagation
all work together to ensure each operation sees its data in the most
efficient arrangement. This is why `SpyreTensorLayout` appears
everywhere in the codebase — it's the thread connecting host tensors
to device computation.

### Key vocabulary recap

| Term | What it is |
|---|---|
| **Stick** | 128 bytes of contiguous, aligned device memory (64 fp16 elements) |
| **Tiling** | Splitting a CPU tensor dimension into stick-sized chunks, adding a device dimension |
| **`SpyreTensorLayout`** | Metadata: `device_size` + `dim_map` + `device_dtype` |
| **`dim_map`** | Maps each device dimension back to a CPU dimension (-1 = synthetic) |
| **`FixedTiledLayout`** | Compiler IR: extends Inductor's `FixedLayout` with `SpyreTensorLayout` |
| **Stickification pass** | Compiler pass that assigns and propagates layouts, inserts restickify ops |
| **Restickify** | Data movement op that reorganizes sticks without changing logical values |
| **DCI** | Data Copy Instruction — the nested-loop DMA spec for host↔device transfer |
| **Sparse tensor** | A tensor with one meaningful element per stick (from reductions) |
| **Padding** | Extra elements added to fill a stick (tracked, masked during computation) |

---

## Chapter 4: Adding a New Operation

> **Dive deeper:**
> [Adding Operations](docs/source/compiler/adding_operations.md) for the
> canonical operation cookbook, and
> [Supported Operations](docs/source/user_guide/supported_operations.md)
> for the current list of supported ops.

This is the practical chapter. By now you understand the compilation
pipeline (Chapter 2) and tensor layouts (Chapter 3). Here we walk
through what you actually do when Spyre needs to support a new
operation.

### The first question: which pattern?

There are three patterns for adding an op, and the choice depends on
one thing: **how does the Spyre hardware implement this operation?**

```text
Can Spyre execute this op directly as a single OpFunc?
  ├─ YES → Pattern 1: Direct mapping (1 file to touch)
  └─ NO
      ├─ Can it be rewritten as a combination of ops Spyre already supports?
      │   └─ YES → Pattern 2: Decomposition (1 file to touch)
      └─ Is it a Spyre-specific fused op with no ATen equivalent?
          └─ YES → Pattern 3: Custom op (3 files to touch)
```

Let's walk through each one with real examples from the codebase.

### Pattern 1: Direct ATen → OpFunc Mapping

**When:** A pointwise ATen op maps directly to a single Spyre hardware
operation (OpFunc).

**What to do:** Add one method to `SpyreOpFuncs` in
`torch_spyre/_inductor/spyre_kernel.py`.

**Real example — `sigmoid`:**

The ATen op `aten.sigmoid` maps directly to Spyre's `"sigmoid"` OpFunc.
The entire change is:

```python
# In SpyreOpFuncs (spyre_kernel.py):
@staticmethod
def sigmoid(x):
    return PointwiseOp("sigmoid", [x])
```

That's it. One method, one file. When Inductor's handler encounters a
`sigmoid` during kernel compilation (Chapter 2, Stage 5), it calls
`SpyreOpFuncs.sigmoid()`, which returns a `PointwiseOp` that gets
assembled into an `OpSpec`.

**Why this works for `sigmoid` and `reciprocal`:** Inductor actually has
*default decompositions* for these ops that break them into simpler math
(e.g., sigmoid = 1/(1+exp(-x))). But by adding a `SpyreOpFuncs` method,
you're telling the compiler "don't decompose this — Spyre has a native
implementation." The direct mapping overrides the default decomposition.

**More examples:** `abs`, `add`, `mul`, `relu`, `exp`, `neg`, `tanh`,
`sqrt`, `rsqrt` — see the full list in `SpyreOpFuncs`.

**For ops with constants** (non-tensor arguments like thresholds or
scaling factors), use `op_info`:

```python
@staticmethod
def clamp(x, min, max):
    op_info = {
        "constants": {
            "clipMin": min,
            "clipMax": max,
        }
    }
    return PointwiseOp("clip", [x], op_info)
```

The `op_info` dict flows through the OpSpec and into the SuperDSC JSON,
where DeepTools reads the constants.

### Pattern 2: Spyre-Specific Decomposition

**When:** The ATen op should be rewritten into other ops that Spyre
already supports, before the graph reaches lowering.

**What to do:** Register a decomposition in
`torch_spyre/_inductor/decompositions.py`.

**Real example — `layer_norm`:**

PyTorch's `aten.layer_norm` is a high-level op that combines mean,
variance, normalization, and scaling. Spyre doesn't have a single
OpFunc for this, but it does have three specialized ops that do the
job when chained together: `exx2` (compute mean/variance),
`layernormscale` (compute normalization factor), and `layernormnorm`
(apply normalization with weight and bias).

```python
# In decompositions.py:
@register_spyre_decomposition([torch.ops.aten.layer_norm.default])
def spyre_layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    mean = torch.ops.spyre.exx2(input, 1.0 / normalized_shape[0], False)
    norm_mean = torch.ops.spyre.layernormscale(mean, eps)
    return torch.ops.spyre.layernormnorm(input, mean, norm_mean, weight, bias)
```

Notice something: the decomposition rewrites `aten.layer_norm` into
`spyre.exx2`, `spyre.layernormscale`, and `spyre.layernormnorm` — these
are **custom Spyre ops** (Pattern 3). The decomposition replaces one
node in the FX Graph with three nodes, each of which the rest of the
pipeline knows how to handle.

As we discussed in Chapter 2, this is a lookup table entry — not a
graph rewrite. When Inductor encounters `aten.layer_norm` during
tracing, it calls this function instead, and traces the three
replacement ops.

**Simpler example — `gelu`:**

```python
@register_spyre_decomposition([torch.ops.aten.gelu.default])
def spyre_gelu(input, approximate="none"):
    return torch.ops.spyre.gelu(input, approximate)
```

This just redirects to a custom Spyre op. The decomposition exists to
intercept the ATen op early and route to the Spyre-specific version.

### Pattern 3: Custom Op + Lowering

**When:** The operation has no ATen equivalent — it's a
Spyre-specific fused or specialized op that the hardware supports but
PyTorch doesn't know about.

**What to do:** Touch three files.

**Step 1: Define the custom op** in
`torch_spyre/_inductor/customops.py`:

```python
@torch.library.custom_op("spyre::gelu", mutates_args=(), device_types="spyre")
def gelu(input: torch.Tensor, approximate: str = "none") -> torch.Tensor:
    # CPU fallback — used when tracing or running outside Spyre
    pass

@gelu.register_fake
def _(input: torch.Tensor, approximate: str = "none"):
    # Shape inference — tells the compiler what shape the output will be.
    # Called during tracing with "fake" (shape-only) tensors.
    return input.new_empty(input.size())
```

Two pieces here:
- The `@custom_op` defines the operation's signature and registers it
  in PyTorch's op registry as `torch.ops.spyre.gelu`. The
  `device_types="spyre"` means it only dispatches on Spyre tensors.
- The `@register_fake` function tells Dynamo/AOTAutograd what output
  shape to expect. It never sees real data — just shapes and dtypes.
  This is needed because the compiler traces with "fake tensors" to
  figure out shapes without running the actual computation.

**Step 2: Register a lowering** in
`torch_spyre/_inductor/lowering.py`:

```python
@register_spyre_lowering(torch.ops.spyre.gelu)
def lower_gelu(x, approximate="none"):
    pw = Pointwise.create(
        device=x.get_device(),
        dtype=x.get_dtype(),
        inner_fn=inner_fn,
        ranges=x.get_size(),
    )
    pw.realize()
    return pw
```

This tells Inductor how to lower `spyre.gelu` into Loop-Level IR.
For pointwise ops, you create a `Pointwise` node; for reductions,
you'd create a `SpyreReduction`.

**Step 3: Add to SpyreOpFuncs** in
`torch_spyre/_inductor/spyre_kernel.py`:

```python
@staticmethod
def gelu(x):
    return PointwiseOp("gelufwd", [x])
```

This maps the op to its Spyre hardware name (`"gelufwd"`) for the
OpSpec → SuperDSC codegen.

**The full chain for a custom op:**

```text
User writes: torch.nn.GELU(x)
  → Dynamo traces: aten.gelu(x)
  → Decomposition intercepts: spyre.gelu(x)
  → Lowering creates: Pointwise IR node
  → SpyreOpFuncs maps: PointwiseOp("gelufwd", [x])
  → OpSpec + SuperDSC: {"opfunc": "gelufwd", ...}
  → DeepTools compiles to binary
```

### Pattern 2.5: Fallback to CPU

Sometimes you don't want to implement an op on Spyre at all — you just
want it to work by falling back to the CPU. This is useful for ops that
are rarely performance-critical or that Spyre doesn't support yet.

**What to do:** Register a fallback in
`torch_spyre/ops/fallbacks.py`:

```python
@register_fallback([aten.sin.default, aten.sin.out])
def spyre__sin(input, **kwargs):
    return torch.sin(input, **kwargs)
```

The `@register_fallback` decorator automatically:
1. Moves all tensor inputs from Spyre to CPU.
2. Runs the operation on CPU.
3. Moves the result back to Spyre.

This is transparent to the user but obviously slower than a native
implementation (two data transfers + CPU execution). Fallback ops
also emit a warning so developers know something isn't running
natively.

### Writing tests

Every new op needs tests. The test infrastructure lives in
`tests/inductor/test_inductor_ops.py` and uses two key utilities from
`tests/inductor/utils_inductor.py`:

**`compare_with_cpu(fn, *args)`** — the workhorse. It:
1. Runs `fn(*args)` on CPU to get a reference result.
2. Compiles `fn` for Spyre and runs it.
3. Optionally runs `fn` in eager mode on Spyre.
4. Compares all results within tolerance (`atol=0.1`, `rtol=0.1` by
   default — generous because float16 is less precise).

```python
def test_my_new_op(self):
    x = cached_randn((64, 128))  # stick-aligned shape
    compare_with_cpu(torch.my_op, x)
```

**`compare(fn, *args)`** — a three-way comparison: compiled Spyre vs.
uncompiled CPU vs. the sendnn backend. Use this when you want to verify
against the lower-level backend too.

**Tests use `ParameterizedTestMeta`** to generate many test cases from
parameter sets. The pattern is:

```python
class TestOps(unittest.TestCase, metaclass=ParameterizedTestMeta):
    PARAMS = {
        ("test_my_op",): {
            "ops_dict": {"my_op": torch.my_op},
            "shapes": [(64,), (64, 128), (5, 100, 150)],
        },
    }

    def test_my_op(self, op, x):
        compare_with_cpu(op, x)
```

The metaclass generates individual test methods like
`test_my_op_shape_64` and `test_my_op_shape_64_128` from the parameter
set. This gives you coverage across 1D–4D shapes, stick-aligned sizes
(multiples of 64), and non-aligned sizes (which exercise padding).

### Quick reference: files to touch

| Pattern | Files | When |
|---|---|---|
| Direct mapping | `spyre_kernel.py` | ATen op → single Spyre OpFunc |
| Decomposition | `decompositions.py` | ATen op → combination of existing ops |
| Custom op | `customops.py` + `lowering.py` + `spyre_kernel.py` | Spyre-specific op with no ATen equivalent |
| Fallback | `ops/fallbacks.py` | Run on CPU for now |
| Tests | `tests/inductor/test_inductor_ops.py` | Always |

### How to figure out which pattern to use

When you're asked to add support for an op (say, `aten.foo`):

1. **Check if Spyre has a native OpFunc for it.** Look at the existing
   `SpyreOpFuncs` methods for similar ops. If Spyre's hardware docs or
   the DeepTools documentation list a matching op, use Pattern 1.

2. **Check if it can be decomposed.** Can `aten.foo` be expressed as a
   combination of ops already in `SpyreOpFuncs`? If yes, use Pattern 2.
   Look at existing decompositions for inspiration.

3. **Check if it needs a custom op.** If neither of the above works —
   for example, Spyre has a specialized fused implementation that's
   more efficient than decomposing — use Pattern 3.

4. **Fallback as a last resort.** If you just need the op to not crash
   and performance doesn't matter, register a CPU fallback.

### Key vocabulary recap

| Term | What it is |
|---|---|
| **Pattern 1** | Direct ATen → OpFunc mapping (one entry in `SpyreOpFuncs`) |
| **Pattern 2** | Spyre-specific decomposition (rewrite into existing supported ops) |
| **Pattern 3** | Custom op (new op definition + lowering + OpFunc entry) |
| **Fallback** | CPU-side execution via `@register_fallback` |
| **`@custom_op`** | PyTorch decorator for registering a new op in the op registry |
| **`@register_fake`** | Shape inference function used during tracing (no real data) |
| **`compare_with_cpu`** | Test utility: compile for Spyre, run on CPU, compare results |

---

## Chapter 5: Eager Mode

> **Dive deeper:**
> [Runtime Overview](docs/source/runtime/overview.md) for the runtime
> layer (device lifecycle, memory allocation, kernel execution), and
> [Debugging Guide](docs/source/user_guide/debugging.md) for
> investigating incorrect behavior and compilation failures.

Chapters 2–4 focused on the compiled path (`torch.compile`). But most
PyTorch code runs without compilation — you just write operations and
they execute immediately. This is **eager mode**, and torch-spyre
supports it too.

### What eager mode means

In eager mode, there's no tracing, no FX Graph, no Inductor, no
SuperDSC. Each operation executes the moment Python encounters it:

```python
x = torch.randn(64, 128, dtype=torch.float16, device="spyre")
y = torch.relu(x)       # executes immediately, returns result
z = y + x                # executes immediately, returns result
```

Every line is a complete round-trip: Python tells the device "do this,"
the device does it, Python gets the result. Simple, debuggable, but
slower than compiled mode because of per-operation overhead.

### How PyTorch dispatches to Spyre

When you call `torch.relu(x)` on a Spyre tensor, PyTorch needs to find
the right implementation. This is the **dispatcher** — PyTorch's routing
system that looks at the operation and the tensor's device, then calls
the correct backend function.

torch-spyre registers its eager ops under the **PrivateUse1** dispatch
key (which was renamed to `"spyre"`). There are three mechanisms used,
depending on the operation:

#### Mechanism 1: Direct kernel registration (`eager.py`)

For operations that need hand-written Spyre implementations, torch-spyre
registers kernels directly in `torch_spyre/ops/eager.py` using
`@torch.library.register_kernel`:

```python
# In eager.py:
@torch.library.register_kernel("aten::fill_.Scalar", ["spyre"])
def spyre__fill_scalar(self, other):
    tmp = torch.ones(self.size(), dtype=self.dtype) * other
    self.copy_(tmp)
    return self
```

This tells PyTorch: "when `aten::fill_.Scalar` is called on a Spyre
tensor, use this function." The dispatcher sees the `"spyre"` device
and routes directly here.

Some of these are simple CPU-side implementations (like `fill_` above,
which creates a CPU tensor and copies it to the device). Others are
more interesting — they use `torch.compile` internally:

```python
@torch.library.register_kernel("aten::mm", ["spyre"])
def spyre__mm(self, mat2):
    compiled_mm = torch.compile(torch.mm, dynamic=False)
    return compiled_mm(self, mat2)
```

This is a key pattern: **eager mode calls compiled mode under the
hood.** When you do `x @ y` on Spyre tensors without `torch.compile`,
the eager dispatch for `mm` actually compiles and runs the operation
through the full Inductor pipeline (Chapter 2). The compilation result
is cached, so subsequent calls with the same shapes reuse the compiled
kernel.

This makes sense for Spyre because the device operates on compiled
programs — there's no way to "just run" an uncompiled matmul on the
hardware. The eager wrapper hides this complexity from the user.

#### Mechanism 2: DispatchKey registration (decompositions → eager)

Remember from Chapter 4 that `@register_spyre_decomposition` registers
decompositions for the compiled path? It also does double duty: for
ATen ops, it simultaneously registers a **PrivateUse1 dispatch kernel**
so the same decomposition works in eager mode too.

Why is this necessary? Some ATen ops have a dispatch key called
**CompositeImplicitAutograd (CIA)** — meaning PyTorch provides a
default implementation that works for *any* backend by decomposing
into simpler ops. The problem is that CIA dispatch happens *before*
the device-specific dispatch. So if `aten.layer_norm` is CIA, calling
it on a Spyre tensor would use PyTorch's generic decomposition (which
may produce ops Spyre can't handle) instead of Spyre's custom
decomposition. Registering a PrivateUse1 kernel explicitly ensures
Spyre's version takes priority over the CIA fallback.

Here's how it works. When you register a decomposition like:

```python
@register_spyre_decomposition([torch.ops.aten.layer_norm.default])
def spyre_layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    mean = torch.ops.spyre.exx2(input, 1.0 / normalized_shape[0], False)
    norm_mean = torch.ops.spyre.layernormscale(mean, eps)
    return torch.ops.spyre.layernormnorm(input, mean, norm_mean, weight, bias)
```

Behind the scenes, `register_spyre_decomposition` also calls
`register_spyre_decompositions_via_dispatchkey`, which creates an
**`OPWrapper`** — an object that decides what to do based on context:

```python
class OPWrapper:
    def __init__(self, op, spyre_fn):
        self.spyre_fn = spyre_fn
        self._compiled_fn = torch.compile(spyre_fn, dynamic=False)

    def __call__(self, *args, **kwargs):
        if torch.compiler.is_compiling():
            # Inside torch.compile: call the decomposition directly
            return self.spyre_fn(*args, **kwargs)
        else:
            # Eager mode: use the pre-compiled version
            return self._compiled_fn(*args, **kwargs)
```

The `OPWrapper` is registered as the PrivateUse1 kernel for the op. So
when `aten.layer_norm` is called on a Spyre tensor in eager mode:

1. The dispatcher finds the PrivateUse1 kernel (the `OPWrapper`).
2. `OPWrapper.__call__` sees we're not inside `torch.compile`.
3. It calls `self._compiled_fn` — a pre-compiled version of the
   decomposition function.
4. That triggers the full compilation pipeline (Dynamo → Inductor →
   Spyre backend → DeepTools) and caches the result.

The pre-compilation (`torch.compile(spyre_fn, dynamic=False)`) happens
once when the wrapper is created. Subsequent calls with the same shapes
skip compilation entirely and reuse the cached kernel.

#### Mechanism 3: CPU fallbacks (`fallbacks.py`)

For ops that Spyre doesn't implement natively, the fallback mechanism
(covered in Chapter 4) provides eager-mode support by bouncing to CPU:

```python
@register_fallback([aten.sin.default, aten.sin.out])
def spyre__sin(input, **kwargs):
    return torch.sin(input, **kwargs)
```

The `@register_fallback` decorator registers a PrivateUse1 kernel that:
1. Moves tensor inputs from Spyre → CPU.
2. Runs the operation on CPU.
3. Moves results back CPU → Spyre.

This is transparent but slow (two DMA transfers + CPU execution).
Fallback ops emit a warning so you know they're not running natively.

### The initialization sequence

These registrations don't all happen at import time. They follow a
specific order, triggered by `_SpyreImpl._lazy_init()` (the first
time you access the Spyre device):

1. **C++ runtime starts** — `torch_spyre._C.start_runtime()` initializes
   the hardware allocator and device.

2. **Tensor patches applied** — `_patch_tensor_for_spyre()` in
   `_monkey_patch.py` adds Spyre-specific behavior to `torch.Tensor`:
   - Custom `__repr__` that shows Spyre tensor contents (copies to CPU
     for display).
   - Extended `.to()` that accepts a `device_layout` argument.
   - Extended `torch.empty()` that accepts a `device_layout` argument.
   - A `.device_tensor_layout()` method to inspect the
     `SpyreTensorLayout`.

3. **Inductor backend registered** — `_autoload()` registers
   `SpyreInterface`, `SuperDSCScheduling`, and
   `SpyrePythonWrapperCodegen` with PyTorch.

4. **Custom ops loaded** — `customops.py` is imported, making
   `torch.ops.spyre.*` available.

5. **DispatchKey kernels registered** —
   `_register_spyre_dispatchkey_kernels_permanently()` takes all the
   `OPWrapper` instances from step 2-style registrations and
   permanently installs them as PrivateUse1 dispatch kernels. The
   `Library` objects are stored in module-level globals so they're
   never garbage-collected (which would silently unregister the
   kernels).

### The big picture: eager = compiled, just hidden

The most important takeaway: **on Spyre, eager mode is largely
compiled mode with the compilation hidden behind dispatch wrappers.**
Unlike a CPU or GPU where eager ops have hand-written C++/CUDA kernels,
Spyre's hardware requires compiled programs. So the eager wrappers
call `torch.compile` under the hood, cache the results, and present
a seamless eager interface.

This means:
- **First call is slow** — it triggers compilation (Dynamo → Inductor →
  DeepTools). Subsequent calls with the same shapes are fast.
- **Same code paths** — a bug in the compiled path usually affects
  eager mode too, because they share the same backend.
- **Some ops are pure CPU** — operations like `fill_`, `normal_`,
  `zero_` create data on CPU and copy it to the device. These don't
  go through the compiler at all.

```text
User calls torch.relu(spyre_tensor)
  → PyTorch dispatcher finds PrivateUse1 kernel
     ├─ Direct kernel (eager.py)?      → hand-written implementation
     ├─ OPWrapper (decompositions.py)? → torch.compile(decomp_fn)(args)
     └─ Fallback (fallbacks.py)?       → move to CPU, run, move back
```

### Key vocabulary recap

| Term | What it is |
|---|---|
| **Dispatcher** | PyTorch's routing system that maps (operation + device) → implementation |
| **PrivateUse1** | The dispatch key used for custom devices; renamed to `"spyre"` |
| **`OPWrapper`** | Object that routes between compiled context (direct call) and eager (pre-compiled) |
| **CIA** | CompositeImplicitAutograd — PyTorch's generic decomposition that custom backends must override |
| **`_lazy_init`** | Thread-safe initialization that runs on first device access |
| **`torch.library`** | PyTorch API for registering kernels and custom ops with the dispatcher |

---

## Chapter 6: Core Division & Work Distribution

> **Dive deeper:**
> [Work Division Planning](docs/source/compiler/work_division_planning.md)
> for the full planning algorithm with diagrams, and
> [Work Division Codegen](docs/source/compiler/work_division_codegen.md)
> for how division plans are translated into executable code.

Spyre has 32 cores. A single operation on a large tensor can't run on
just one core and expect good performance — it would leave 31 cores
idle. This chapter explains how the compiler divides work across cores.

### The basic idea

Core division is conceptually simple: if you have a [1024, 128] tensor
and 32 cores, you could give each core 32 rows (1024 / 32 = 32). Each
core runs the same program on its slice of the data — this is called
**SPMD** (Single Program, Multiple Data).

The tricky parts:
- Not all dimensions divide evenly by 32.
- Some operations have multiple dimensions you could split.
- Some dimensions (like the reduction dimension in a matmul) have
  different splitting semantics than output dimensions.
- The split must respect stick boundaries.

### Where core division happens in the pipeline

Core division runs as a **scheduler pass** — after lowering and
stickification, but before kernel compilation (Chapter 2, Stage 4).
The entry point is `core_division_planning()` in
`torch_spyre/_inductor/core_division.py`.

```text
Loop-Level IR nodes (with layouts assigned by stickification)
    │
    ▼
core_division_planning()
    ├─ For each Pointwise node  → divide_pointwise_op()
    ├─ For each Reduction node  → divide_reduction_op()
    └─ Others                   → single-core (no division)
    │
    ▼
Nodes annotated with op_dim_splits and n_cores_used
```

The pass walks the scheduler graph in topological order. For each
operation, it decides how to split the work, then annotates the node.
These annotations flow into the OpSpec (Chapter 2, Stage 5) and
ultimately into the SuperDSC JSON that tells each core which slice
of data to process.

### The core splitting algorithm

The algorithm is a **greedy, priority-based approach**:

1. List the dimensions that can be parallelized, with their sizes.
2. Assign priorities (which dimension to split first).
3. Starting with the highest-priority dimension, find the largest
   divisor of its size that doesn't exceed the remaining core budget.
4. Allocate those cores, reduce the budget, move to the next dimension.
5. Stop when cores are exhausted or no even divisions remain.

The constraint that splits must **divide evenly** is important — Spyre
doesn't support uneven splits (no "core 31 gets fewer elements than the
rest"). If 32 doesn't divide your dimension size evenly, the algorithm
finds the largest divisor that does.

Here's the actual core function:

```python
def core_split(size: int, max_cores: int) -> int:
    """Find the largest divisor of size that doesn't exceed max_cores."""
    for i in range(max_cores, 0, -1):
        if size % i == 0:
            return i
    return 1
```

And the multi-dimensional version:

```python
def multi_dim_core_split(sizes, max_cores, priorities):
    splits = [1] * len(sizes)
    # Sort dimensions by priority (highest first)
    # For each dimension, greedily allocate cores
    for dim_idx, size, _ in sorted_by_priority:
        best_split = core_split(size, remaining_cores)
        splits[dim_idx] = best_split
        remaining_cores //= best_split
    return splits
    # Product of all splits = total cores used
```

### Pointwise operations

For element-wise ops (add, relu, mul, etc.), every output element is
independent — perfect for parallelism. The strategy is straightforward:
split the output tensor's dimensions across cores.

**Example:** `relu` on a [128, 256] tensor with 32 cores:

```text
sizes      = [128, 256]     # but 256 is in sticks, so really [128, 4]
priorities = [128, 4]       # larger dimensions get higher priority

core_split: 128 / 32 = 4 → splits[0] = 32, remaining = 1
Result: op_dim_splits = [32, 1]
→ 32 cores, each processing 4 rows of 256 elements
```

**Important detail:** for the stick dimension (the innermost device
dimension), the parallelizable unit is the number of *sticks*, not
elements. A 256-element row at fp16 is 4 sticks. You can't split
within a stick — the hardware reads whole sticks.

**Limitation:** if any input has a different shape (broadcasting), core
division is skipped and the op runs on a single core. This is a current
limitation, not a fundamental constraint.

### Matrix multiplication

Matmul is more interesting because it has three dimensions: M (rows),
K (reduction), and N (columns) in `C[M,N] = A[M,K] × B[K,N]`.

The priority order matters:

| Priority | Dimension | Why |
|---|---|---|
| Highest (3) | M (rows) | Output dimension, independent slices |
| Medium (2) | N (columns) | Output dimension, independent slices |
| Lowest (1) | K (reduction) | Splitting the reduction creates partial sums that must be accumulated — more complex |

**Example:** 8 cores, M=128, K=64, N=64:

```text
sizes      = [128, 64, 64]     # [M, K, N]
priorities = [3, 1, 2]          # M first, then N, then K

Step 1: M=128, try core_split(128, 8) → 8. But then no cores for N.
        Actually: core_split(128, 8) → 4. splits[M] = 4, remaining = 2
Step 2: K=64, but N has higher priority, so N goes first.
        core_split(64, 2) → 2. splits[N] = 2, remaining = 1
Result: op_dim_splits = [4, 1, 2]   # [M_split, K_split, N_split]
→ 8 cores total: 4 row tiles × 2 column tiles
→ Each core computes a 32×32 block of the output
```

K is only split as a last resort because splitting the reduction
dimension means each core computes a *partial* sum, and the partial
sums need to be combined afterward — more complexity in the backend.

### Batched matrix multiplication

BMM adds batch dimensions. For a 3D `[B, M, K] × [B, K, N]`:

| Priority | Dimension | Why |
|---|---|---|
| Highest (4) | B (batch) | Completely independent — perfect parallelism |
| High (3) | N (columns) | Output dimension |
| Medium (2) | M (rows) | Output dimension |
| Lowest (1) | K (reduction) | Partial sums, avoid if possible |

Batch dimensions get top priority because each batch is entirely
independent — no partial sums, no accumulation, no communication
between cores.

### How the splits reach the hardware

After `core_division_planning` annotates each node, the information
flows through the remaining pipeline stages:

1. **OpSpec** (Stage 5) — `op_info["n_cores_used"]` and
   `op_info["op_dim_splits"]` are stored in the OpSpec.

2. **SDSCSpec** (Stage 6) — `parse_op_spec` converts splits into
   `work_slices` (how many slices per dimension) and
   `core_id_to_work_slice` (which slice each core gets). This uses
   modular arithmetic to map core IDs to multi-dimensional slice
   coordinates:

   ```python
   # For op_dim_splits = [4, 1, 2] with 8 cores:
   # core 0 → slice (0, 0, 0)  — row tile 0, col tile 0
   # core 1 → slice (0, 0, 1)  — row tile 0, col tile 1
   # core 2 → slice (1, 0, 0)  — row tile 1, col tile 0
   # ...
   # core 7 → slice (3, 0, 1)  — row tile 3, col tile 1
   ```

3. **SuperDSC JSON** — the `coreIdToWkSlice_` field tells each core
   exactly which slice of the iteration space it owns.

4. **DeepTools** — reads the per-core slice assignments and generates
   binary code where each core processes its assigned data region.

### Configuring the core count

The number of cores is controlled by the `SENCORES` environment
variable (default: 32, valid range: 1–32):

```bash
SENCORES=1 python my_script.py   # Single-core execution (useful for debugging)
SENCORES=8 python my_script.py   # Use 8 cores
SENCORES=32 python my_script.py  # Full parallelism (default)
```

Setting `SENCORES=1` disables core division entirely — useful for
isolating bugs that might be caused by the division logic.

### Key vocabulary recap

| Term | What it is |
|---|---|
| **SPMD** | Single Program, Multiple Data — all cores run the same program on different data |
| **`op_dim_splits`** | Per-dimension split counts (e.g., [4, 1, 2] = 4×1×2 = 8 cores) |
| **Core split** | Finding the largest divisor of a dimension that fits the core budget |
| **Work slice** | The specific region of iteration space assigned to one core |
| **Stick boundary** | Splits respect stick alignment — can't split within a 64-element stick |

---

## Chapter 7: End-to-End Walkthrough

We've covered every stage individually. Now let's trace a single
example through the entire pipeline — from Python code to device
execution — and see concrete data at each step.

### The example

```python
import torch
import torch_spyre

@torch.compile
def model(x, w):
    return torch.relu(x @ w)

x = torch.randn(32, 64, dtype=torch.float16, device="spyre")
w = torch.randn(64, 128, dtype=torch.float16, device="spyre")
y = model(x, w)
```

Two operations: a matrix multiply (`x @ w`) producing a [32, 128]
result, then a pointwise relu. Let's follow them through every stage.

### Stage 1: Dynamo traces the FX Graph

Dynamo watches `model` execute and captures:

```text
%x       = placeholder             # [32, 64] fp16, device=spyre
%w       = placeholder             # [64, 128] fp16, device=spyre
%mm      = call_function: aten.mm(%x, %w)
%relu    = call_function: aten.relu(%mm)
return (%relu,)
```

Four nodes, two operations. Dynamo doesn't know about Spyre — this is
the same graph you'd get for CPU or GPU. The `compile_fx` wrapper
(Chapter 2) detects Spyre tensors in the inputs and activates the
Spyre context.

### Stage 2: AOTAutograd

Strips gradient tracking. Since this is inference with no
`requires_grad`, the graph passes through unchanged.

### Stage 3a: Decompositions

Inductor checks the decomposition table for `aten.mm` and `aten.relu`.

- **`aten.mm`** — no Spyre-specific decomposition registered. It stays
  as-is.
- **`aten.relu`** — no Spyre-specific decomposition either. It stays
  as-is.

Both ops are simple enough that Spyre supports them directly. If this
were `aten.layer_norm`, the decomposition would expand it into three
custom Spyre ops here (Chapter 4, Pattern 2).

### Stage 3b: FX Graph Passes

`CustomPostPasses` runs. The passes look for patterns like scalar ops
to convert to tensor ops, or `mm` patterns to convert to `bmm`. Our
graph has a plain 2D `mm` and a `relu` — neither triggers a rewrite.
The graph is unchanged.

### Stage 3c: Lowering — FX Graph → Loop-Level IR

Each FX node is lowered to Loop-Level IR.

**`aten.mm` → Reduction:**

```text
Reduction(
    ranges          = [32, 128],      # output shape: M=32 rows, N=128 cols
    reduction_ranges = [64],          # K=64 (dimension being summed)
    inner_fn         = (idx, r_idx) → (x[idx[0], r_idx[0]],
                                       w[r_idx[0], idx[1]]),
    reduction_type   = "matmul"
)
```

**`aten.relu` → Pointwise:**

```text
Pointwise(
    ranges   = [32, 128],            # same shape as mm output
    inner_fn = (idx) → relu(mm_result[idx[0], idx[1]])
)
```

### Stage 4a: Scheduler graph construction

Inductor organizes these into a scheduler graph:

```text
SchedulerNode "buf0" (Reduction: matmul)
    reads:  x, w
    writes: buf0        ← the mm result [32, 128]
        │
        ▼
SchedulerNode "buf1" (Pointwise: relu)
    reads:  buf0
    writes: buf1        ← the relu result [32, 128]
```

`buf1` depends on `buf0` — relu can't run until matmul finishes.

### Stage 4b: No fusion

`SuperDSCScheduling.can_fuse_vertical(buf0, buf1)` returns `False`.
Each stays as its own kernel. On a GPU, these would likely be fused
into one kernel. On Spyre, they stay separate so DeepTools can optimize
each independently.

### Stage 4c: Stickification — assign tensor layouts

The stickification pass walks the graph and assigns `SpyreTensorLayout`
to every buffer.

**Input `x` [32, 64]:**

64 elements in the last dimension = exactly 1 stick. No tiling needed
for the stick dimension. The first dimension (32) gets tiled.

```text
SpyreTensorLayout(
    device_size = [1, 32, 64],     # 1 tile, 32 rows, 64 elems/stick
    dim_map     = [1, 0, 1],       # device dim 0 → CPU dim 1 (cols, tiled)
                                   # device dim 1 → CPU dim 0 (rows)
                                   # device dim 2 → CPU dim 1 (cols, within stick)
    device_dtype = SEN169_FP16
)
```

**Input `w` [64, 128]:**

128 elements in the last dimension = 2 sticks. Tiling splits it.

```text
SpyreTensorLayout(
    device_size = [2, 64, 64],     # 2 tiles, 64 rows, 64 elems/stick
    dim_map     = [1, 0, 1],       # cols tiled into 2 groups of 64
    device_dtype = SEN169_FP16
)
```

**Output `buf0` (mm result) [32, 128]:**

The matmul output layout is derived from the input layouts. The output
has N=128 columns → 2 sticks, M=32 rows.

```text
SpyreTensorLayout(
    device_size = [2, 32, 64],     # 2 tiles, 32 rows, 64 elems/stick
    dim_map     = [1, 0, 1],
    device_dtype = SEN169_FP16
)
```

**Output `buf1` (relu result) [32, 128]:**

Relu is pointwise — its output inherits the same layout as its input
(`buf0`). Same `SpyreTensorLayout` as above. No restickify needed
because the layouts match.

### Stage 4d: Core division

With `SENCORES=32` (default):

**Matmul** (`buf0`): dimensions are M=32, K=64, N=128.

But core division works in parallelizable units — for the stick
dimension, that's the number of sticks, not elements. So:

```text
M = 32 (rows)
K = 1  (sticks in reduction dim: 64/64 = 1)
N = 2  (sticks in output cols: 128/64 = 2)

priorities = [3, 1, 2]    # M highest, then N, then K
```

Step 1: M=32, `core_split(32, 32)` → 32. But 32 × 2 = 64 > 32.
So: `core_split(32, 16)` → 16. splits[M] = 16, remaining = 2.
Step 2: N=2, `core_split(2, 2)` → 2. splits[N] = 2, remaining = 1.

```text
op_dim_splits = [16, 1, 2]    # [M_split, K_split, N_split]
n_cores_used  = 32            # 16 × 1 × 2 = 32
```

Each core computes a 2×64 block of the output (2 rows, 1 stick of 64
columns).

**Relu** (`buf1`): pointwise on [32, 128].

```text
sizes = [32, 2]    # 32 rows, 2 sticks
priorities = [32, 2]

core_split(32, 32) → 32. But 32 × 2 = 64 > 32.
core_split(32, 16) → 16. splits[0] = 16, remaining = 2.
core_split(2, 2) → 2. splits[1] = 2, remaining = 1.

op_dim_splits = [16, 2]
n_cores_used  = 32
```

Same 32 cores, same 2-row × 1-stick blocks. Because the relu has the
same shape and layout as the matmul output, the division happens to
match — each core processes the same region of data in both kernels.

### Stage 5: Kernel compilation — Loop-Level IR → OpSpec

`SpyreKernel` processes each scheduled node.

**Matmul kernel:**

The handler replays the Reduction IR. The `inner_fn` loads pairs of
elements from `x` and `w`. The `store_reduction` method sees a
`ReductionOp("matmul", [x_access, w_access])` and builds:

```text
OpSpec(
    op              = "matmul",
    is_reduction    = True,
    iteration_space = [32, 64, 128],     # [M, K, N]
    args            = [
        TensorArg(is_input=True,  ...),  # x [32, 64]
        TensorArg(is_input=True,  ...),  # w [64, 128]
        TensorArg(is_input=False, ...),  # buf0 [32, 128] (output)
    ],
    op_info = {
        "n_cores_used": 32,
        "op_dim_splits": [16, 1, 2],
    }
)
```

**Relu kernel:**

The handler replays the Pointwise IR. `SpyreOpFuncs.relu(x)` returns
`PointwiseOp("relufwd", [x])`. The `store` method builds:

```text
OpSpec(
    op              = "relufwd",
    is_reduction    = False,
    iteration_space = [32, 128],         # [M, N]
    args            = [
        TensorArg(is_input=True,  ...),  # buf0 [32, 128]
        TensorArg(is_input=False, ...),  # buf1 [32, 128] (output)
    ],
    op_info = {
        "n_cores_used": 32,
        "op_dim_splits": [16, 2],
    }
)
```

### Stage 6a: OpSpec → SDSCSpec → SuperDSC JSON

`parse_op_spec` converts each OpSpec to an SDSCSpec.

**Matmul SDSCSpec** (simplified):

```text
SDSCSpec(
    opfunc           = "matmul",
    execution_unit   = "pt",              # matrix unit (not scalar/vector)
    data_format      = SEN169_FP16,
    num_cores        = 32,
    iteration_space  = {mb: 32, in: 64, out: 128},
    work_slices      = {mb: 16, in: 1, out: 2},
    core_id_to_work_slice = {
        mb: core_id % 16,                # which row tile
        in: 0,                            # no K split
        out: floor(core_id / 16) % 2,    # which col tile
    },
    ...
)
```

Core 0 gets row tile 0, col tile 0 → rows 0–1, cols 0–63.
Core 1 gets row tile 1, col tile 0 → rows 2–3, cols 0–63.
...
Core 16 gets row tile 0, col tile 1 → rows 0–1, cols 64–127.
...
Core 31 gets row tile 15, col tile 1 → rows 30–31, cols 64–127.

`generate_sdsc` renders this into the final JSON:

```json
{
    "matmul": {
        "numCoresUsed_": 32,
        "dscs_": [{
            "matmul": {
                "N_": {"mb_": 32, "in_": 64, "out_": 128},
                "numWkSlicesPerDim_": {"mb": 16, "in": 1, "out": 2},
                "coreIdToWkSlice_": {
                    "0": {"mb": 0, "in": 0, "out": 0},
                    "1": {"mb": 1, "in": 0, "out": 0},
                    ...
                    "16": {"mb": 0, "in": 0, "out": 1},
                    ...
                },
                ...
            }
        }]
    }
}
```

The relu kernel gets a similar JSON, but with `opfunc: "relufwd"`,
`execution_unit: "sfp"` (scalar/vector unit), and a 2D iteration space.

### Stage 6b: Host-side wrapper code

`SpyrePythonWrapperCodegen` generates a Python file that looks roughly
like:

```python
# Auto-generated host code (simplified)
from torch_spyre._C import spyre_empty_with_layout, SpyreTensorLayout
from torch_spyre.execution.async_compile import SpyreAsyncCompile

async_compile = SpyreAsyncCompile()

# Compile both kernels (can happen in parallel)
matmul_kernel = async_compile.spyre(matmul_sdsc_json, ...)
relu_kernel   = async_compile.spyre(relu_sdsc_json, ...)

def call(args):
    x, w = args

    # Allocate intermediate buffer for matmul output
    buf0 = spyre_empty_with_layout(
        (32, 128), (128, 1), torch.float16,
        SpyreTensorLayout([2, 32, 64], [1, 0, 1], ..., SEN169_FP16)
    )

    # Allocate output buffer for relu
    buf1 = spyre_empty_with_layout(
        (32, 128), (128, 1), torch.float16,
        SpyreTensorLayout([2, 32, 64], [1, 0, 1], ..., SEN169_FP16)
    )

    # Launch kernels in sequence
    matmul_kernel(x, w, buf0)     # x @ w → buf0
    relu_kernel(buf0, buf1)        # relu(buf0) → buf1

    return (buf1,)
```

This is real, executable Python. The first call triggers DeepTools
compilation of the two JSON specs into device binaries. Subsequent calls
reuse the compiled binaries.

### Stage 7: DeepTools → Binary → Execution

DeepTools takes each SuperDSC JSON and produces optimized binary
programs. For the matmul, it maps the per-core work slices onto Spyre's
matrix multiplication dataflow hardware. For the relu, it maps the
pointwise operation onto the scalar/vector processing units.

At runtime:
1. DMA transfers move `x` and `w` from host to device DDR (reorganizing
   from row-major to tiled stick layout).
2. The matmul binary executes — all 32 cores process their assigned
   2×64 blocks in parallel, reading from `x` and `w`, writing to `buf0`.
3. The relu binary executes — all 32 cores apply relu to their assigned
   blocks of `buf0`, writing to `buf1`.
4. DMA transfers move `buf1` back from device to host (reorganizing from
   tiled stick layout back to row-major).
5. The result tensor `y` is returned to Python.

### The complete journey, in one picture

```text
Python: y = torch.relu(x @ w)
    │
    ▼
Dynamo: FX Graph [mm, relu] ──────────── 2 ATen op nodes
    │
    ▼
Decompositions: unchanged ─────────────── both ops supported directly
    │
    ▼
FX Passes: unchanged ──────────────────── no patterns to rewrite
    │
    ▼
Lowering: Loop-Level IR ───────────────── Reduction(matmul) + Pointwise(relu)
    │
    ▼
Scheduler: 2 nodes, no fusion ─────────── each op is its own kernel
    │
    ▼
Stickify: assign layouts ──────────────── [32,128] → device [2, 32, 64]
    │
    ▼
Core division: 32 cores each ──────────── matmul [16,1,2], relu [16,2]
    │
    ▼
SpyreKernel: 2 OpSpecs ────────────────── "matmul" + "relufwd"
    │
    ▼
Codegen: 2 SuperDSC JSONs ─────────────── per-core work assignments
       + host Python ──────────────────── alloc, compile, launch, return
    │
    ▼
DeepTools: 2 device binaries ──────────── optimized for Spyre silicon
    │
    ▼
Execution: DMA in → matmul → relu → DMA out → result in Python
```

### What to notice

- **Two operations became two kernels.** On a GPU these would likely
  fuse into one. On Spyre they stay separate — DeepTools handles
  optimization at a lower level.

- **The same 32-core division applies to both kernels.** Because the
  relu has the same shape as the matmul output, each core works on the
  same region of data in both kernels. This isn't always the case —
  if the shapes differed, core division would be computed independently.

- **Stickification added no restickify ops.** The matmul output layout
  and the relu input layout are compatible. If we'd added a transpose
  between them, the stickification pass would have inserted a
  restickify operation to reorganize the sticks.

- **The host code is straightforward Python.** Allocate buffers, launch
  kernels in sequence, return. All the complexity is in the JSON specs
  and the compiled binaries.

- **First call pays the compilation cost.** Dynamo tracing, Inductor
  lowering, SuperDSC generation, and DeepTools compilation all happen
  on the first invocation. The second call with the same shapes skips
  everything and just runs the cached binaries.

---

*This concludes the core education guide. For further exploration, see
the linked documentation in each chapter's "Dive deeper" section, or
the skills in `.claude/skills/` for task-specific guidance.*