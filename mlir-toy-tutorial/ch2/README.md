# Chapter 2 — Emitting Basic MLIR

**Tutorial:** https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/

---

## What This Chapter Is About

In Chapter 1 we built a tree (AST) representing our Toy program. In Chapter 2 we **walk that AST and convert it into MLIR** — the actual intermediate representation used by the compiler.

This chapter introduces the most fundamental MLIR concepts: what a dialect is, what an operation is, how types work, and how to programmatically build MLIR using `OpBuilder`.

---

## What is MLIR?

MLIR stands for **Multi-Level Intermediate Representation**. It is a framework for building compilers. Instead of one fixed IR (like LLVM IR), MLIR lets you define your own custom IR called a **dialect**.

Think of MLIR as a container format. Inside it, different dialects coexist. Our Toy compiler defines a `toy` dialect with Toy-specific operations.

---

## What is a Dialect?

A dialect is a named namespace that groups together:
- **Operations** (instructions, like toy.transpose)
- **Types** (data types, like tensors)
- **Attributes** (constant values attached to ops)

We register our dialect like this:
```cpp
context.getOrLoadDialect<ToyDialect>();
```

After Chapter 2, our MLIR output uses the `toy` dialect exclusively. Later chapters will mix it with other dialects (`affine`, `memref`, `llvm`) as we lower.

---

## What Operations Did We Define?

We defined these operations in the `toy` dialect using **ODS (TableGen)**:

| Operation | What it does |
|-----------|-------------|
| `toy.constant` | Stores a constant tensor value |
| `toy.add` | Element-wise addition of two tensors |
| `toy.mul` | Element-wise multiplication of two tensors |
| `toy.transpose` | Transposes a tensor (flips rows and columns) |
| `toy.reshape` | Changes the shape of a tensor |
| `toy.print` | Prints a tensor to stdout |
| `toy.func` | Defines a Toy function |
| `toy.return` | Returns from a Toy function |
| `toy.generic_call` | Calls a user-defined Toy function |

---

## What is ODS?

ODS stands for **Operation Definition Specification**. Instead of writing C++ boilerplate for every operation by hand, we write a `.td` (TableGen) file and MLIR auto-generates the C++ code.

For example, the transpose op is declared as:
```tablegen
def TransposeOp : Toy_Op<"transpose"> {
  let summary = "transpose operation";
  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor);
}
```

From this one declaration, MLIR generates: the C++ class, the builder method, the verifier, the printer, and the parser. This saves a huge amount of code.

---

## What is MLIRGen?

MLIRGen is our code that walks the AST from Chapter 1 and emits MLIR operations. For each AST node it visits, it creates the corresponding MLIR operation using `OpBuilder`.

For example, when it visits a `BinaryExprAST` with operator `*`:
```cpp
// It calls:
builder.create<MulOp>(location, lhs, rhs);
// Which creates: toy.mul %lhs, %rhs : tensor<*xf64>
```

---

## What are Types at This Stage?

At this point, tensors are **unranked** — written as `tensor<*xf64>`. This means we know it's a tensor of 64-bit floats, but we don't know the shape yet (2x3? 3x2? something else?).

Shape inference (knowing the actual dimensions) happens in Chapter 4. For now, everything is `tensor<*xf64>`.

---

## Key Source Files

| File | What it does |
|------|-------------|
| `include/toy/Ops.td` | ODS declarations for all 9 Toy operations |
| `include/toy/Dialect.h` | C++ declaration of the ToyDialect class |
| `mlir/Dialect.cpp` | Registers the dialect, implements op behavior |
| `mlir/MLIRGen.cpp` | Walks AST nodes and emits MLIR using OpBuilder |
| `toyc.cpp` | Driver: parse → MLIRGen → print MLIR |

---

## The Command Run

```cmd
toyc-ch2.exe codegen.toy -emit=mlir
```

What this does step by step:
1. Lexer + Parser build the AST (same as Ch1)
2. MLIRGen walks the AST and creates MLIR operations for each node
3. `-emit=mlir` tells the compiler: "stop here and print the MLIR"
4. The MLIR module is printed

See `output.txt` for the full MLIR output.

---

## What the Output Means

In `output.txt` you will see:
```mlir
module {
  toy.func @multiply_transpose(%arg0: tensor<*xf64>, %arg1: tensor<*xf64>) -> tensor<*xf64> {
    %0 = toy.transpose(%arg0 : tensor<*xf64>) to tensor<*xf64>
    %1 = toy.transpose(%arg1 : tensor<*xf64>) to tensor<*xf64>
    %2 = toy.mul %0, %1 : tensor<*xf64>
    toy.return %2 : tensor<*xf64>
  }
  toy.func @main() {
    %0 = toy.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
    ...
    toy.print %4 : tensor<*xf64>
    toy.return
  }
}
```

Reading this:
- `toy.func @multiply_transpose(...)` — the Toy function we defined
- `%arg0: tensor<*xf64>` — first argument, unranked tensor (shape unknown)
- `%0 = toy.transpose(...)` — result of transpose stored in SSA value `%0`
- `toy.return %2` — returns the result
- `dense<[[1.0, 2.0, 3.0], ...]>` — a constant tensor with literal values
- `toy.generic_call @multiply_transpose(...)` — calling our function

Everything is in **SSA (Static Single Assignment)** form — each value (like `%0`, `%1`, `%2`) is assigned exactly once and never changed.
