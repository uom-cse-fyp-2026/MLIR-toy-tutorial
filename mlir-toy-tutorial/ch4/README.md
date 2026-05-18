# Chapter 4 — Interfaces: Shape Inference and Inlining

**Tutorial:** https://mlir.llvm.org/docs/Tutorials/Toy/Ch-4/

---

## What This Chapter Is About

Chapter 4 solves two important problems:

1. **Inlining** — the built-in MLIR inliner doesn't know anything about Toy. How do we make it work with our dialect?
2. **Shape Inference** — all our tensors are still `tensor<*xf64>` (unknown shape). Before we can lower to real memory operations, we need to know the actual dimensions.

The answer to both problems is **Interfaces** — a way to plug dialect-specific knowledge into generic MLIR passes without modifying those passes.

---

## Problem 1: Inlining

**What is inlining?**
Inlining means replacing a function call with the body of the function directly. For example:

```toy
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}
def main() {
  var c = multiply_transpose(a, b);  ← function call
}
```

After inlining:
```toy
def main() {
  var c = transpose(a) * transpose(b);  ← body copied in, call removed
}
```

This is important because after inlining, the optimizer can see the full picture and apply more optimizations.

**Why doesn't the inliner work automatically?**
MLIR has a generic built-in inliner pass, but it doesn't know:
- How to handle `toy.return` when inlining (what do you do with the return value?)
- Whether Toy functions are safe to inline
- How to copy the function body into the call site

We need to tell it these things through an **Interface**.

**The solution — InlinerInterface:**
We implement `ToyInlinerInterface` which inherits from `DialectInlinerInterface`. It tells the inliner:
- Yes, all Toy functions are legal to inline
- When you see `toy.return`, replace the call's result with the returned value
- How to handle the function boundaries during inlining

---

## Problem 2: Shape Inference

**Why do we need shape inference?**
In Chapter 2, all tensors were `tensor<*xf64>` — the shape was unknown. But to generate real code (memory allocations, loop bounds), we need concrete shapes like `tensor<3x2xf64>`.

After inlining, we can trace where values come from. We know `a` is `tensor<2x3xf64>` from its constant definition. So `transpose(a)` must be `tensor<3x2xf64>`. And `transpose(a) * transpose(a)` must also be `tensor<3x2xf64>`. This is shape inference — propagating known shapes forward.

**The solution — ShapeInferenceOpInterface:**
We define a new interface that requires each op to implement one method:
```
inferShapes() — look at my input shapes and set my output shape
```

Each op implements this differently:
- `TransposeOp.inferShapes()` → output shape is input shape with dimensions reversed
- `MulOp.inferShapes()` → output shape is same as input shapes (element-wise)
- `AddOp.inferShapes()` → output shape is same as input shapes

**The Shape Inference Pass:**
We write a pass that:
1. Collects all ops that still have unknown output shapes
2. For each such op, if all its inputs have known shapes, calls `inferShapes()`
3. Repeats until all shapes are known

---

## The Combined Result

After running Chapter 4 with `-opt` (which runs inlining then shape inference), the output shows:

**Before (Ch2 MLIR):**
```mlir
%4 = toy.generic_call @multiply_transpose(%1, %3)
     : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
toy.print %4 : tensor<*xf64>
```

**After (Ch4 MLIR with -opt):**
```mlir
%1 = toy.transpose(%0 : tensor<2x3xf64>) to tensor<3x2xf64>
%2 = toy.mul %1, %1 : tensor<3x2xf64>
toy.print %2 : tensor<3x2xf64>
```

Notice:
- The `toy.generic_call` is gone — the function was inlined
- All shapes are now concrete (`tensor<3x2xf64>` instead of `tensor<*xf64>`)
- The IR is now ready for lowering to actual memory operations

---

## Key Source Files

| File | What it does |
|------|-------------|
| `include/toy/ShapeInferenceInterface.td` | ODS declaration of the ShapeInference interface |
| `mlir/ShapeInferencePass.cpp` | The pass that applies shape inference iteratively |
| `mlir/Dialect.cpp` | Registers ToyInlinerInterface with the dialect |
| `include/toy/Ops.td` | Ops declare they implement ShapeInferenceOpInterface |

---

## The Command Run

```cmd
toyc-ch4.exe codegen.toy -emit=mlir -opt
```

What this does:
1. Lexer + Parser → AST
2. MLIRGen → MLIR
3. `-opt` runs: inliner pass → shape inference pass
4. `-emit=mlir` prints the resulting MLIR

See `output.txt` — all tensors have concrete shapes and no function calls remain.

---

## Why Interfaces Are Powerful

The key insight is that the MLIR inliner knows nothing about Toy, and the Toy dialect knows nothing about the inliner's internals. But through the `DialectInlinerInterface`, they can cooperate perfectly.

This is the same design used by MLIR's other generic passes (canonicalization, CSE, dead code elimination). Any dialect can plug into them by implementing the right interface — without modifying the passes themselves. This makes MLIR extremely extensible.
