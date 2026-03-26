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
| `torch_spyre/csrc/` | C++ code that talks to the actual hardware (memory, tensors, kernel execution). |
| `torch_spyre/ops/` | Eager-mode operation implementations and CPU fallbacks. |
| `torch_spyre/device/` | PyTorch device interface (how PyTorch discovers and talks to Spyre). |
| `tests/` | Test suites for eager ops, compiled ops, and building blocks. |
| `docs/` | Architecture docs, user guides, and reference material. |

### How it all connects (the life of a tensor operation)

Here's what happens when you do `z = x + y` on a Spyre device:

**In eager mode:**
1. PyTorch's dispatcher sees the operation targets a `"spyre"` device.
2. It routes to the Spyre implementation registered for `add`.
3. That implementation calls into the C++ layer (`torch_spyre._C`), which
   talks to the hardware, does the math, and returns a new Spyre tensor.

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
  creates separate graphs for each segment.
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

After lowering, the graph is a collection of Loop-Level IR nodes. The
**scheduler** decides how to group them into **kernels** — units of work
that will be sent to the device as a single program.

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
   Spyre's on-chip scratchpad memory (a small, fast memory close to the
   compute units). Only runs when `LX_PLANNING=1`.

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
    device_dtype: DataFormats  # on-device data format (e.g. SEN169_FP16)
    device_size: list[int]     # shape in device memory (stick-aligned)
    device_coordinates: list   # how iteration space maps to device dimensions
    dtype: torch.dtype         # host-side dtype (e.g. torch.float16)
    it_dim_map: list[int]      # maps each iteration dimension to a tensor dimension
                               #   (-1 = broadcast/reduction for this tensor)
    allocation: Any            # scratchpad memory offset, if applicable
    device_layout: SpyreTensorLayout  # full device memory layout descriptor
```

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
tensors** (tensors where reductions produce only one meaningful element
per stick, with the rest being padding).

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

---

*Next: Chapter 5 could cover Eager Mode (how single operations dispatch
without compilation), Core Division & Work Distribution (how computation
is split across 32 cores), or the C++ Layer (torch_spyre/csrc/ and the
hardware runtime).*