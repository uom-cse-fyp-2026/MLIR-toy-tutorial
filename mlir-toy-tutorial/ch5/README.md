# Chapter 5 - Partial Lowering to Affine Dialect

**Tutorial:** https://mlir.llvm.org/docs/Tutorials/Toy/Ch-5/

## What This Chapter Does
Lowers Toy operations to lower-level MLIR dialects: `affine`, `memref`, and `arith`.

## Command Used

```bash
toyc-ch5.exe codegen.toy -emit=mlir-affine
```
## Output
See `output.txt` — Toy tensor ops become explicit `affine.for` loops over `memref` buffers. `toy.print` is kept for Ch6.

## Lowering Map
| Toy Op | Lowered To |
|--------|-----------|
| toy.add | affine.for + arith.addf |
| toy.mul | affine.for + arith.mulf |
| toy.transpose | affine.for + memref.load/store |
| toy.func | func.func |

## Key Concepts
- **DialectConversion**: Framework for lowering between dialects
- **ConversionPattern**: Pattern that replaces one dialect's op with another's
- **TypeConverter**: Maps tensor types to memref types
- **affine.for**: Loop with statically known bounds enabling loop optimizations