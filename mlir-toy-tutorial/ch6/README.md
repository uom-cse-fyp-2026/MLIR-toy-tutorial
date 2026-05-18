# Chapter 6 - Lowering to LLVM IR & Code Generation

**Tutorial:** https://mlir.llvm.org/docs/Tutorials/Toy/Ch-6/

## What This Chapter Does
Completes the lowering pipeline all the way to LLVM IR.

## Command Used

```bash
toyc-ch6.exe codegen.toy -emit=llvm
```

## Output
See `output.txt` for the full LLVM IR output including malloc, printf calls, and loop structures.

## Full Pipeline

Toy dialect → Affine + memref + arith → LLVM dialect → LLVM IR

## Key Concepts
- **LLVMTypeConverter**: Converts MLIR types to LLVM types
- **translateModuleToLLVMIR**: Converts MLIR llvm dialect to llvm::Module
- **toy.print** is lowered to `printf` calls
- **LLJIT**: LLVM's JIT compiler for execution

