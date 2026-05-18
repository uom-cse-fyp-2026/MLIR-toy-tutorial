# MLIR Toy Tutorial - All 7 Chapters

This repository contains my implementation and study of the [MLIR Toy Tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/).

The Toy Tutorial walks through building a complete compiler for a simple tensor-based language called **Toy**, using **MLIR (Multi-Level Intermediate Representation)** from the LLVM project.

## What is MLIR?
MLIR is a compiler framework that allows defining custom dialects (IR abstractions), writing transformations, and lowering high-level code step by step down to LLVM IR and machine code.

## Environment
- OS: Windows
- LLVM built from source: `llvm-project`
- Build system: CMake + Ninja

## Chapter Overview

| Chapter | Topic |
|---------|-------|
| Ch1 | Toy Language & AST |
| Ch2 | Emitting Basic MLIR |
| Ch3 | High-level Optimizations (Pattern Rewriting) |
| Ch4 | Interfaces: Shape Inference & Inlining |
| Ch5 | Partial Lowering to Affine Dialect |
| Ch6 | Lowering to LLVM IR & Code Generation |
| Ch7 | Adding a Custom Struct Type |

## How to Run
Each chapter has its own `output.txt` showing the actual output when the chapter binary was run.