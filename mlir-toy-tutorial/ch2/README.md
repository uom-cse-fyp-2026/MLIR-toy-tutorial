# Chapter 2 - Emitting Basic MLIR

**Tutorial:** https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/

## What This Chapter Does
Defines a custom `toy` MLIR dialect and walks the AST to emit MLIR operations using OpBuilder.

## Command Used

```bash
toyc-ch2.exe codegen.toy -emit=mlir
```
## Output
See `output.txt` for the MLIR dialect output.

## Key Concepts
- **Dialect**: Named namespace grouping operations and types
- **Operations**: toy.constant, toy.add, toy.mul, toy.transpose, toy.print, toy.func, toy.return
- **ODS (TableGen)**: Declarative way to define ops
- **OpBuilder**: Used to create and insert MLIR operations
- Tensors are unranked `tensor<*xf64>` at this stage