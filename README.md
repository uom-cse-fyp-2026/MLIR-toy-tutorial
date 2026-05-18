# MLIR Toy Tutorial — All 7 Chapters

**Tutorial Source:** https://mlir.llvm.org/docs/Tutorials/Toy/

---

## What is This Tutorial About?

This tutorial teaches you how to build a **complete compiler from scratch** using **MLIR (Multi-Level Intermediate Representation)**, which is part of the LLVM project.

To make it easier to learn, the tutorial uses a simple made-up language called **Toy**. Toy is a small tensor-based language — meaning every value is a tensor (like a matrix or array). It supports basic math operations, user-defined functions, and printing results.

The goal is NOT to learn the Toy language. The goal is to use Toy as a vehicle to learn how **real compilers are built using MLIR** — the same infrastructure used by Google, Apple, AMD, and others to build compilers for AI/ML frameworks, GPUs, and specialized hardware.

---

## What Problem Does MLIR Solve?

Before MLIR, if you wanted to build a compiler for a new language or hardware target, you had two bad choices:

1. **Write everything yourself from scratch** — very hard, takes years
2. **Use LLVM directly** — but LLVM only has one level of IR (low-level), so you lose all your high-level information (shapes, types, semantics) very early

MLIR fixes this by letting you define **multiple levels of IR** (called dialects), and gradually lower from high-level to low-level step by step — keeping useful information at each level for optimization.

**Real world example:** TensorFlow and PyTorch both use MLIR internally to compile neural networks down to GPU or CPU code efficiently.

---

## What is the Toy Language?

```toy
# Define a function
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}

# Main entry point
def main() {
  var a<2, 3> = [[1, 2, 3], [4, 5, 6]];  # 2x3 tensor
  var b<2, 3> = [1, 2, 3, 4, 5, 6];
  var c = multiply_transpose(a, b);
  print(c);
}
```

Key features of Toy:
- Every value is a **64-bit floating point tensor**
- Supports **user-defined functions**
- Operations: `+`, `*`, `transpose()`, `print()`
- Shape can be declared explicitly `<2, 3>` or inferred automatically
- Intentionally minimal so we can focus on compiler concepts, not language features

---

## The Big Picture — Full Compiler Pipeline

When you run a Toy program, it goes through these transformation stages across all 7 chapters:

```
Your .toy source file
        │
        ▼
┌───────────────┐
│  Chapter 1    │  LEXER + PARSER
│               │  Source text → tokens → Abstract Syntax Tree (AST)
└──────┬────────┘
       │
       ▼
┌───────────────┐
│  Chapter 2    │  MLIR GENERATION (MLIRGen)
│               │  AST → MLIR "toy dialect" with ops like toy.transpose, toy.mul
└──────┬────────┘
       │
       ▼
┌───────────────┐
│  Chapter 3    │  OPTIMIZATION (Pattern Rewriting)
│               │  transpose(transpose(x)) → x   (eliminated completely)
└──────┬────────┘
       │
       ▼
┌───────────────┐
│  Chapter 4    │  INTERFACES (Inlining + Shape Inference)
│               │  Function calls inlined, tensor<*xf64> → tensor<3x2xf64>
└──────┬────────┘
       │
       ▼
┌───────────────┐
│  Chapter 5    │  PARTIAL LOWERING (Toy → Affine + memref + arith)
│               │  tensor ops → explicit affine.for loops over memory buffers
└──────┬────────┘
       │
       ▼
┌───────────────┐
│  Chapter 6    │  FULL LOWERING (→ LLVM IR + JIT Execution)
│               │  malloc, printf, loops in LLVM IR → actual program runs!
└──────┬────────┘
       │
       ▼
┌───────────────┐
│  Chapter 7    │  EXTENSION (Custom Struct Type)
│               │  Add struct composite type to the Toy type system
└───────────────┘
        │
        ▼
   Output printed to screen
```

---

## Environment & Setup

| Item | Detail |
|------|--------|
| OS | Windows |
| LLVM Source | `D:\llvm-project` |
| Build Folder | `D:\llvm-project\build\bin\` |
| Build System | CMake + Ninja |
| Binaries Built | `toyc-ch1.exe` through `toyc-ch7.exe` |

---

## Chapter Overview

| Chapter | Topic | Key Concept |
|---------|-------|-------------|
| [Ch1](./ch1/) | Toy Language & AST | Lexer, Parser, AST |
| [Ch2](./ch2/) | Emitting Basic MLIR | Dialects, Ops, ODS, OpBuilder |
| [Ch3](./ch3/) | High-Level Optimizations | Pattern Rewriting, Canonicalization |
| [Ch4](./ch4/) | Interfaces | Inlining, Shape Inference, OpInterface |
| [Ch5](./ch5/) | Partial Lowering | DialectConversion, Affine, memref |
| [Ch6](./ch6/) | Full Lowering to LLVM | LLVM IR, JIT, printf |
| [Ch7](./ch7/) | Custom Struct Type | TypeDef, Custom Types |

---

## The `-emit` Flag Explained

Each chapter binary accepts a `-emit` flag that controls how far the compiler goes:

| Flag | What it outputs | Used In |
|------|----------------|---------|
| `-emit=ast` | Abstract Syntax Tree dump | Ch1 |
| `-emit=mlir` | MLIR toy dialect IR | Ch2, Ch3, Ch4, Ch7 |
| `-emit=mlir -opt` | MLIR after optimizations | Ch3, Ch4 |
| `-emit=mlir-affine` | After lowering to Affine dialect | Ch5 |
| `-emit=llvm` | Final LLVM IR | Ch6 |
| `-emit=jit` | Compiles and runs the program | Ch6 |

---

## Key MLIR Concepts Learned

**Dialect** — A named namespace grouping operations and types. We built a `toy` dialect. MLIR has many built-in dialects: `affine`, `memref`, `arith`, `llvm`.

**Operation (Op)** — Every instruction in MLIR is an operation with inputs, outputs, and attributes. Example: `toy.transpose(%x : tensor<2x3xf64>) to tensor<3x2xf64>`.

**ODS (Operation Definition Specification)** — A TableGen-based way to declaratively define ops. MLIR auto-generates the C++ boilerplate from `.td` files.

**Lowering** — Translating from high-level IR to lower-level IR. We lowered: `toy` → `affine+memref` → `llvm` → machine code.

**Pattern Rewriting** — Rules like "if you see X, replace it with Y". MLIR applies them automatically. Used for optimization.

**Interface** — Lets generic MLIR passes (like the inliner) work with any dialect. The dialect implements the interface to answer the pass's questions.

**Shape Inference** — Figuring out the concrete shape of every tensor. Starts with unknown `tensor<*xf64>`, ends with concrete `tensor<3x2xf64>`.

---

## What This Tutorial Teaches

After completing this tutorial, you understand:

1. How a real compiler pipeline works end-to-end — source text to machine code
2. How to define your own IR dialect in MLIR with custom operations and types
3. How to write optimization passes using pattern rewriting
4. How interfaces allow dialect-agnostic transformations like inlining
5. How to progressively lower IR through multiple abstraction levels
6. How LLVM IR is structured and how MLIR generates it
7. How to extend MLIR's type system with custom composite types

---

*Based on the official MLIR Toy Tutorial: https://mlir.llvm.org/docs/Tutorials/Toy/*
