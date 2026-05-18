# Chapter 4 - Interfaces: Shape Inference & Inlining

**Tutorial:** https://mlir.llvm.org/docs/Tutorials/Toy/Ch-4/

## What This Chapter Does
Plugs the Toy dialect into MLIR's generic inliner and adds shape inference so all tensors get concrete shapes.

## Command Used

```bash
toyc-ch4.exe codegen.toy -emit=mlir -opt
```
## Output
See `output.txt` — functions are inlined and all tensors now have concrete shapes like `tensor<3x2xf64>` instead of `tensor<*xf64>`.

## Key Concepts
- **InlinerInterface**: Tells MLIR's inliner how to handle Toy functions
- **ShapeInferenceOpInterface**: Each op implements `inferShapes()` to propagate shapes
- **ShapeInferencePass**: Worklist algorithm that propagates shapes until convergence