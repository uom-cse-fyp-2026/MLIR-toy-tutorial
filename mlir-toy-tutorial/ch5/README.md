# Chapter 5 — Partial Lowering to Affine and Standard Dialects

**Tutorial:** https://mlir.llvm.org/docs/Tutorials/Toy/Ch-5/

---

## What This Chapter Is About

Up to Chapter 4, we worked entirely in the `toy` dialect — a high-level representation that closely matches what the programmer wrote. Now we start **lowering**: translating from the high-level `toy` dialect down to lower-level dialects that are closer to how a real CPU actually works.

This is called **partial lowering** because we lower most Toy ops but intentionally leave `toy.print` for Chapter 6.

---

## What Does "Lowering" Mean?

Lowering means replacing high-level abstract operations with more concrete, lower-level ones.

**High-level thinking:** "multiply two tensors element-wise"
**Low-level reality:** "allocate memory, loop over every element, multiply each pair, store the result"

The high-level view is easier for humans and optimization. The low-level view is what hardware actually executes. Lowering bridges this gap step by step.

---

## What Dialects Do We Lower To?

| Dialect | Purpose |
|---------|---------|
| `affine` | Mathematical loop abstractions (`affine.for`, `affine.load`, `affine.store`) |
| `memref` | Memory buffer type and operations (`memref.alloc`, `memref.dealloc`) |
| `arith` | Basic arithmetic (`arith.addf`, `arith.mulf`, `arith.constant`) |
| `func` | Standard function definitions (`func.func`, `func.return`) |

---

## How Does Each Toy Op Get Lowered?

### toy.transpose → affine.for loops

`toy.transpose` on a `tensor<2x3xf64>` becomes:
1. Allocate a new buffer: `memref.alloc() : memref<3x2xf64>`
2. Two nested loops over the dimensions
3. Load from `[i, j]` in the source, store to `[j, i]` in the destination

```mlir
%alloc = memref.alloc() : memref<3x2xf64>
affine.for %i = 0 to 2 {
  affine.for %j = 0 to 3 {
    %val = affine.load %source[%i, %j] : memref<2x3xf64>
    affine.store %val, %alloc[%j, %i] : memref<3x2xf64>
  }
}
```

### toy.mul → affine.for + arith.mulf

`toy.mul` element-wise multiplication becomes nested loops with `arith.mulf` inside:
```mlir
affine.for %i = 0 to 3 {
  affine.for %j = 0 to 2 {
    %a = affine.load %lhs[%i, %j] : memref<3x2xf64>
    %b = affine.load %rhs[%i, %j] : memref<3x2xf64>
    %result = arith.mulf %a, %b : f64
    affine.store %result, %output[%i, %j] : memref<3x2xf64>
  }
}
```

### toy.constant → arith.constant values stored into memref

### toy.func → func.func (standard function)

### toy.print → kept as-is (lowered in Ch6)

---

## What is the Affine Dialect and Why Use It?

The `affine` dialect is special because its loops have **statically analyzable bounds** — the loop bounds are known at compile time and expressed as mathematical affine expressions.

This enables powerful automatic optimizations like:
- **Loop tiling** — break loops into smaller tiles for cache efficiency
- **Loop fusion** — merge separate loops into one
- **Vectorization** — convert loops to SIMD instructions

If we went directly to raw C-style loops, these optimizations would be much harder to apply.

---

## How Does Dialect Conversion Work?

MLIR's `DialectConversion` framework manages the lowering process:

1. **ConversionTarget** — declares what is legal at the destination level. We say: "after this pass, no `toy` ops should remain (except toy.print)"
2. **TypeConverter** — tells MLIR how to convert types. `tensor<2x3xf64>` → `memref<2x3xf64>`
3. **ConversionPatterns** — one pattern per op being lowered. Each pattern gets an op and produces lower-level ops
4. **applyPartialConversion()** — runs all patterns, verifies all targeted ops were lowered

---

## Key Source Files

| File | What it does |
|------|-------------|
| `mlir/LowerToAffineLoops.cpp` | All ConversionPatterns + the LowerToAffine pass |
| `include/toy/Passes.h` | Pass interface declarations |
| `toyc.cpp` | Adds `-emit=mlir-affine` option |

---

## The Command Run

```cmd
toyc-ch5.exe codegen.toy -emit=mlir-affine
```

What this does:
1. Lexer + Parser → AST
2. MLIRGen → MLIR toy dialect
3. LowerToAffineLoops pass → converts toy ops to affine/memref/arith
4. `-emit=mlir-affine` prints the result

See `output.txt` — you will see `affine.for` loops, `memref.alloc`, `arith.mulf` etc. The `toy.print` is the only remaining Toy op.

---

## What the Output Shows

In `output.txt`:
- `func.func @main()` — `toy.func` was lowered to a standard function
- `memref.alloc() : memref<3x2xf64>` — memory allocated for result tensor
- `affine.store %cst_4, %alloc_6[0, 0]` — constant values stored into memory
- `affine.for %arg0 = 0 to 3 { affine.for %arg1 = 0 to 2 { ... } }` — the transpose loop
- `arith.mulf %0, %0 : f64` — the element-wise multiply
- `toy.print %alloc : memref<3x2xf64>` — still a Toy op, not lowered yet
- `memref.dealloc` — memory freed after use

---

## Why Is This Called "Partial" Lowering?

Because `toy.print` is intentionally kept in the Toy dialect. This is done on purpose to show that MLIR allows **mixing dialects** — you don't have to lower everything at once. The `toy.print` will be lowered to `printf` calls in Chapter 6.
