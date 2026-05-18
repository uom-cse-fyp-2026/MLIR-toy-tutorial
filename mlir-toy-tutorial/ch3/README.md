# Chapter 3 - High-Level Optimizations (Pattern Rewriting)

**Tutorial:** https://mlir.llvm.org/docs/Tutorials/Toy/Ch-3/

## What This Chapter Does
Implements optimization passes using MLIR's pattern rewriting system.

## Command Used

```bash
toyc-ch3.exe transpose_transpose.toy -emit=mlir -opt
```
## Output
See `output.txt` — notice that `transpose(transpose(x))` is completely eliminated, returning `x` directly.

## Optimizations
- `transpose(transpose(x))` → `x`
- Redundant reshape elimination

## Key Concepts
- **OpRewritePattern**: Base class for writing optimization patterns
- **PatternRewriter**: Safely replaces/erases ops
- **applyPatternsAndFoldGreedily**: Repeatedly applies patterns until no more changes