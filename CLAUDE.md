# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

torch-spyre is an **out-of-tree PyTorch backend** that registers the
**IBM Spyre AI Accelerator** as a first-class PyTorch device (`"spyre"`)
via the PrivateUse1 mechanism. It supports both eager and compiled
(`torch.compile`) execution paths.

## Conventions

- **License:** Apache 2.0 — every source file must carry the 14-line Python
  header (or C++ `/* */` equivalent). See any file in `torch_spyre/` for the
  template.
- **Style:** Google Python Style Guide, Google C++ Style Guide.
- **Imports:** Use `import regex` (aliased as `re` when needed), **never**
  `import re`. A pre-commit hook enforces this.
- **Commits:** Sign off every commit with `git commit -s` (DCO). A pre-commit
  hook auto-appends sign-off.
- **Linting:** Run `pre-commit run --all-files` before pushing. Hooks include
  ruff, clang-format, cpplint, mypy, pymarkdown, yamlfmt, and a custom
  `enforce_regex_import.py`.
- **Line length:** 88 characters (ruff).

## Build and Test

Building requires the IBM Spyre Software Stack (internal).

```bash
# Run all tests
python3 -m pytest tests/

# Run a specific test suite
python3 -m pytest tests/inductor/test_inductor_ops.py

# Run a single test by name
python3 -m pytest tests/inductor/test_inductor_ops.py -k "test_unary_op"

# Run a single test by full path
python3 -m pytest tests/inductor/test_inductor_ops.py::TestOps::test_unary_op

# List available models/cases (custom pytest options)
python3 -m pytest tests/ --list-models
python3 -m pytest tests/ --list-cases

# Run pre-commit checks
pre-commit run --all-files
```

Test sub-suites:

| Suite | Path |
|---|---|
| Eager ops | `tests/test_ops.py` |
| Compiled ops | `tests/inductor/test_inductor_ops.py` |
| Building blocks | `tests/inductor/test_building_blocks.py` |
| Decompositions | `tests/inductor/test_inductor_decomp.py` |
| FX passes | `tests/inductor/test_inductor_fx_passes.py` |
| Tensor layout | `tests/tensor/` |

### Test Utilities

Tests in `tests/inductor/` use a custom framework in `tests/inductor/utils_inductor.py`:

- **`ParameterizedTestMeta`** — metaclass for parameterized test generation.
  Tests declare parameter lists and the metaclass generates individual test
  methods.
- **`compare_with_cpu()`** — compiles and runs an op on both CPU and Spyre,
  then compares results with tolerance.
- **`cached_randn()` / `unique_randn_along_dim()`** — deterministic tensor
  generation utilities.

## Key Environment Variables

| Variable | Purpose |
|---|---|
| `TORCH_SPYRE_DEBUG=1` | Enable C++ debug logging and `-O0` builds |
| `SENCORES` | Number of Spyre cores (1-32, default 32) |
| `LX_PLANNING=1` | Enable LX scratchpad memory planning |
| `TORCH_LOGS="+inductor"` | Verbose Inductor logging |
| `TORCH_COMPILE_DEBUG=1` | Dump Inductor debug artifacts |
| `TORCH_SPYRE_DOWNCAST_WARN=0` | Suppress float32->float16 warnings |

## Spyre Hardware Basics

- Default dtype: `torch.float16`
- **Stick:** 128-byte aligned memory chunk = 64 elements at fp16
- Device name constant: `torch_spyre.constants.DEVICE_NAME` = `"spyre"`
- Up to 32 cores per accelerator, >300 TOPS at 75W

## Architecture

<<<<<<< Updated upstream
See `docs/source/` for detailed architecture documentation:

- `docs/source/architecture/spyre_accelerator.md` — Spyre accelerator overview
- `docs/source/compiler/architecture.md` — compilation pipeline
- `docs/source/user_guide/tensors_and_layouts.md` — tiled tensor layout specification
- `docs/source/compiler/adding_operations.md` — how to add new operations
- `docs/source/compiler/work_division_planning.md` — multi-core work division
=======
### Initialization Chain

1. PyTorch discovers the backend via the `torch.backends` entry point in
   `pyproject.toml`, calling `torch_spyre._autoload()`.
2. `_autoload()` renames PrivateUse1 to `"spyre"`, registers the device
   module, and calls `_inductor._light_autoload()` to wrap `compile_fx`.
3. On first device access, `_SpyreImpl._lazy_init()` loads the C++ extension
   (`torch_spyre._C`), starts the runtime, patches tensors for custom layouts,
   and registers eager dispatch kernels.

### Compilation Pipeline (torch.compile)

```text
User code with torch.compile
  -> Dynamo (traces to FX graph)
  -> AOTAutograd (strips autograd)
  -> Inductor (generates LoopLevelIR)
  -> Spyre backend (registered via SuperDSCScheduling)
     Front-end (Python, in torch_spyre/_inductor/):
       FX passes (passes.py)
       -> Spyre decompositions (decompositions.py)
       -> LoopLevelIR lowering (lowering.py)
       -> OpSpec generation + tiling (spyre_kernel.py, stickify.py)
       -> SuperDSC JSON codegen (codegen/superdsc.py)
     Back-end (proprietary DeepTools):
       -> SDSC/KTIR compiled binaries
```

Key Inductor integration files:

| File | Role |
|---|---|
| `_inductor/__init__.py` | Registers backend, wraps `compile_fx` to inject decompositions |
| `_inductor/dsc.py` | `SuperDSCScheduling` — Inductor scheduler for Spyre kernels |
| `_inductor/wrapper.py` | `SpyrePythonWrapperCodegen` — host-side code generation |
| `_inductor/lowering.py` | LoopLevelIR lowering with Spyre type rules |
| `_inductor/spyre_kernel.py` | Kernel compilation: LoopLevelIR -> OpSpec (tiling, work division) |
| `_inductor/codegen/superdsc.py` | JSON spec generation for backend compiler |
| `_inductor/stickify.py` | Tensor layout planning (128-byte "stick" alignment) |
| `_inductor/core_division.py` | Multi-core work distribution |
| `device/interface.py` | `SpyreInterface` for Dynamo/AOTAutograd integration |
| `device/op_overrides.py` | `SpyreDeviceOpOverrides` for codegen |

### Module Structure

- **`torch_spyre/`** — top-level: initialization, constants, monkey-patches,
  streams
- **`torch_spyre/ops/`** — eager mode ops (`eager.py`) and host fallbacks
  (`fallbacks.py`)
- **`torch_spyre/_inductor/`** — compiled path: decompositions, lowering,
  scheduling, codegen (~25 modules)
- **`torch_spyre/device/`** — PyTorch device interface and op overrides
- **`torch_spyre/execution/`** — async compilation and kernel execution
- **`torch_spyre/csrc/`** — C++ extension (device runtime, memory, tensor
  layouts)

### Detailed Documentation

- `docs/source/compiler/architecture.md` — compilation pipeline
- `docs/source/compiler/tensor_layouts.md` — tiled tensor layout spec
- `docs/source/compiler/adding_operations.md` — how to add new operations
- `docs/source/compiler/work_division_planning.md` — multi-core work division
- `docs/source/spyre.md` — Spyre accelerator overview
>>>>>>> Stashed changes

## Skills

Task-specific guidance is available in `.claude/skills/`. These cover:

- **project-overview** — repo layout, Spyre architecture, compilation pipeline
- **add-spyre-operation** — patterns for adding new ops
- **write-spyre-op-test** — compiled-path op test framework and patterns
- **pr-review** — PR review checklist
- **debug-compilation** — troubleshooting compilation failures
- **write-rfc** — design proposal workflow (RFCs now at https://github.com/torch-spyre/rfcs)

### Writing SKILL.md Files

When creating or editing `.claude/skills/*/SKILL.md` files:

- **YAML frontmatter `description`:** Use a quoted single-line string, not
  a multi-line `>-` block scalar. Pymarkdown does not understand YAML
  frontmatter and will mangle indented continuation lines.
  - Good: `description: "One line describing the skill."`
  - Bad: `description: >-` followed by indented lines
- **Python templates in skills:** Add `# noqa: F401` to imports that are
  only used in commented-out example code, so ruff does not remove them.
