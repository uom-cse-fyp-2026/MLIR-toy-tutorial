# Chapter 8 — Extending the Toy Dialect: `toy.sin` Operator

**UOM CSE FYP 2026 | Compiler-Directed Adaptive Quantization Framework for Edge LLMs**

## Objective

This chapter extends the MLIR Toy dialect by:

1. **Adding a new `toy.sin` operator** with element-wise sine semantics
2. **Adding a lowering pass** that pattern-matches `toy.sin` and replaces it with a 6-term Taylor series approximation using `affine`, `arith`, and `memref` dialect operations
3. **Writing a test case** (`sin.toy`) demonstrating correctness across scalar, 1D, and 2D tensor inputs

---

## What `toy.sin` does

```
%r2 = toy.sin %r1 : ty -> ty,  for ty in {f64, f64^N, f64^(MxN)}
```

For every element: `r2[i,j] = sin(r1[i,j])`

Approximated using the first 6 terms of the Taylor series:

```
sin(x) = x - x^3/6 + x^5/120 - x^7/5040 + x^9/362880 - x^11/39916800
```

This matches the assignment: sum(n=0..5) [(-1)^n / (2n+1)!] * x^(2n+1)

---

## Files changed vs Chapter 6

| File | What changed |
|---|---|
| `include/toy/Ops.td` | Added `SinOp` declaration with Pure + ShapeInference traits |
| `mlir/Dialect.cpp` | Added `SinOp::build()` and `SinOp::inferShapes()` |
| `mlir/MLIRGen.cpp` | Intercept `sin(x)` calls and emit `toy.sin` op |
| `mlir/LowerToAffineLoops.cpp` | Added `SinOpLowering` with 6-term Taylor series |
| `sin.toy` | New test file: 1D vector, 2D matrix, and compound expression |

---

## How to build and run

From `D:\llvm-project\build`:

```powershell
cmake --build . --target toyc-ch8

# See toy.sin in dialect IR
bin\toyc-ch8.exe ..\mlir\examples\toy\Ch8\sin.toy -emit=mlir -opt

# See Taylor series expanded into affine loops
bin\toyc-ch8.exe ..\mlir\examples\toy\Ch8\sin.toy -emit=mlir-affine -opt

# JIT run and see actual sin values printed
bin\toyc-ch8.exe ..\mlir\examples\toy\Ch8\sin.toy -emit=jit -opt
```

---
## Output
<img width="333" height="93" alt="image" src="https://github.com/user-attachments/assets/e821449e-3f4e-4a8c-846b-521e0d79ca54" />

---


## Why this matters for the FYP

The pattern here — declare op in Ops.td, intercept in MLIRGen, write a lowering
pass — is exactly what TVM relay passes do for quantization transformations.
