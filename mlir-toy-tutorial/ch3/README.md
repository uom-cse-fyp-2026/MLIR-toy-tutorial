# Chapter 3 — High-Level Optimizations (Pattern Rewriting)

**Tutorial:** https://mlir.llvm.org/docs/Tutorials/Toy/Ch-3/

---

## What This Chapter Is About

In Chapter 2 we generated MLIR from our Toy program. But that MLIR may contain **redundant or unnecessary operations** that a smart compiler should eliminate. Chapter 3 teaches us how to write **optimization passes** using MLIR's pattern rewriting system.

No new dialect operations are added here. Instead, we teach the compiler how to recognize and simplify certain patterns in the existing MLIR.

---

## Why Do We Need Optimization?

Consider this Toy program:
```toy
def transpose_transpose(x) {
  return transpose(transpose(x));
}
```

Mathematically, transposing a matrix twice gives you back the original matrix. So `transpose(transpose(x))` is exactly the same as just `x`. A smart compiler should detect this and remove both transpose operations entirely.

Without optimization, the MLIR looks like:
```mlir
%0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
%1 = toy.transpose(%0 : tensor<*xf64>) to tensor<*xf64>
toy.return %1 : tensor<*xf64>
```

After optimization with `-opt`, it becomes:
```mlir
toy.return %arg0 : tensor<*xf64>
```

Both transposes are completely gone. The function just returns its input directly.

---

## What is Pattern Rewriting?

Pattern rewriting is MLIR's mechanism for defining optimization rules. You write a pattern that says:

> "If you see **this** structure in the IR, replace it with **this simpler** structure."

MLIR then automatically searches the IR for matching patterns and applies the replacements repeatedly until nothing more can be simplified.

---

## Optimization 1: Transpose of Transpose Elimination

The rule is: `transpose(transpose(x))` → `x`

The pattern in C++ works like this:
1. Look at the current `toy.transpose` operation
2. Check if its input also comes from another `toy.transpose`
3. If yes — replace the outer transpose with the original input of the inner transpose
4. Both transposes become unused and are deleted

```
Before:
  %0 = toy.transpose(%input)
  %1 = toy.transpose(%0)      ← outer transpose
  use(%1)

After:
  use(%input)                 ← directly use original input
  (both transposes deleted)
```

---

## Optimization 2: Redundant Reshape Elimination

The rule is: if you reshape a tensor to a shape, then reshape it again to the same shape, the second reshape is pointless.

```
Before:
  %0 = toy.reshape(%x) to <2x3>
  %1 = toy.reshape(%0) to <2x3>  ← same shape again, pointless

After:
  %0 = toy.reshape(%x) to <2x3>  ← only one reshape needed
```

---

## How Are Patterns Written in Code?

Each pattern is a C++ class inheriting from `OpRewritePattern<T>` where `T` is the op type to match:

```cpp
struct TransposeTransposeOptPattern : public mlir::OpRewritePattern<TransposeOp> {
  mlir::LogicalResult matchAndRewrite(
      TransposeOp op, mlir::PatternRewriter &rewriter) const override {

    // Step 1: Check if input came from another TransposeOp
    auto transposeInput = op.getOperand().getDefiningOp<TransposeOp>();
    if (!transposeInput)
      return mlir::failure();  // Pattern doesn't match, do nothing

    // Step 2: Replace this op with the inner transpose's original input
    rewriter.replaceOp(op, {transposeInput.getOperand()});
    return mlir::success();  // Pattern matched and was applied
  }
};
```

---

## How Does MLIR Apply Patterns?

Patterns are collected into a `RewritePatternSet` and applied with `applyPatternsAndFoldGreedily()`. This function:
1. Scans all operations in the IR
2. Tries every pattern on every op
3. If a pattern matches and rewrites, marks affected ops for re-checking
4. Repeats until no patterns fire anymore (fixed point reached)

This is called **greedy pattern application** — it keeps applying patterns until the IR stops changing.

---

## Key Source Files

| File | What it does |
|------|-------------|
| `mlir/ToyCombine.cpp` | Defines the TransposeTranspose and Reshape patterns |
| `include/toy/Ops.td` | Ops declare `hasCanonicalizer = 1` to register patterns |
| `mlir/Dialect.cpp` | Registers canonicalization patterns with the ops |
| `toyc.cpp` | Adds the `-opt` flag to enable the optimization pass |

---

## The Command Run

```cmd
toyc-ch3.exe transpose_transpose.toy -emit=mlir -opt
```

What this does:
1. Lexer + Parser → AST
2. MLIRGen → MLIR (same as Ch2)
3. `-opt` triggers the optimization pass → patterns are applied
4. `-emit=mlir` prints the optimized MLIR

See `output.txt` for the full output showing the optimized IR.

---

## What the Output Shows

The `output.txt` shows the IR **after** optimization. You can see that the `transpose_transpose` function body is reduced to just a return statement — the two transpose operations were eliminated completely.

This demonstrates a core principle of compiler optimization: **the more information the IR preserves about the program's intent, the more powerful optimizations become possible.** High-level Toy IR lets us see "this is a transpose" — low-level assembly would never let us detect or remove it.
