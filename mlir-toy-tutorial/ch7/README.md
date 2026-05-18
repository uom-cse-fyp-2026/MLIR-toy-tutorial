# Chapter 7 — Adding a Composite Type (Struct)

**Tutorial:** https://mlir.llvm.org/docs/Tutorials/Toy/Ch-7/

---

## What This Chapter Is About

Chapters 1–6 built a complete compiler pipeline. Chapter 7 is an **extension** that shows how to add a brand new feature to the language and compiler: a **struct (composite) type**.

This chapter answers the question: "How do you add a completely new type to MLIR's type system?" It shows that MLIR's type system is extensible — you are not limited to built-in types.

---

## The New Language Feature: Struct

We extend the Toy language with structs — a way to group multiple tensors together under one name:

```toy
# Declare a struct type with two tensor fields
struct Struct {
  var a<2, 3>;
  var b<2, 3>;
}

# Use the struct in a function
def multiply_transpose(value: Struct) {
  return transpose(value.a) * transpose(value.b);
}

def main() {
  # Create a struct value
  var val = Struct {
    a = [[1, 2, 3], [4, 5, 6]],
    b = [[1, 2, 3], [4, 5, 6]]
  };
  var c = multiply_transpose(val);
  print(c);
}
```

This struct has two fields — `a` and `b` — each of which is a tensor. The struct is passed to a function and its fields are accessed with `.a` and `.b`.

---

## What Had to Change?

Adding a struct type touches every layer of the compiler:

**1. Parser (frontend)** — must recognize `struct` declarations, struct literal syntax `Struct { a = ..., b = ... }`, and field access syntax `value.a`

**2. MLIRGen** — must generate MLIR for struct values and field accesses

**3. MLIR Dialect** — two new operations needed:
- `toy.struct_constant` — stores a constant struct value
- `toy.struct_access` — accesses one field by index

**4. Type System** — a brand new `!toy.struct<...>` type must be defined

**5. Shape Inference** — `toy.struct_access` must implement `inferShapes()` to propagate the field's shape

**6. Lowering** — struct ops must be lowered through affine → LLVM like everything else

---

## How is the Custom Type Defined?

The `StructType` is defined in ODS using `TypeDef`:

```tablegen
def Toy_StructType : Toy_Type<"Struct", "struct"> {
  let summary = "Toy struct type";
  let parameters = (ins
    ArrayRefParameter<"::mlir::Type">:$elementTypes
  );
  let assemblyFormat = "`<` $elementTypes `>`";
}
```

This generates a C++ class `StructType` with:
- A `get(context, elementTypes)` factory method
- A `getElementTypes()` accessor
- Automatic print/parse: `!toy.struct<tensor<2x3xf64>, tensor<2x3xf64>>`

---

## The Two New Operations

### toy.struct_constant

Holds a constant struct value with literal tensor values for each field:

```mlir
%0 = toy.struct_constant [
  dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>,
  dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
] : !toy.struct<tensor<*xf64>, tensor<*xf64>>
```

### toy.struct_access

Accesses one field of a struct by its index (0-based):

```mlir
%field_a = toy.struct_access %struct_val[0]
           : !toy.struct<tensor<*xf64>, tensor<*xf64>> -> tensor<*xf64>
```

Field index 0 → `a`, field index 1 → `b`.

---

## What the Output Shows

In `output.txt` (from `-emit=mlir`):

```mlir
toy.func private @multiply_transpose(%arg0: !toy.struct<tensor<*xf64>, tensor<*xf64>>) -> tensor<*xf64> {
  %0 = toy.struct_access %arg0[0] : !toy.struct<...> -> tensor<*xf64>
  %1 = toy.transpose(%0 : tensor<*xf64>) to tensor<*xf64>
  %2 = toy.struct_access %arg0[1] : !toy.struct<...> -> tensor<*xf64>
  %3 = toy.transpose(%2 : tensor<*xf64>) to tensor<*xf64>
  %4 = toy.mul %1, %3 : tensor<*xf64>
  toy.return %4 : tensor<*xf64>
}
toy.func @main() {
  %0 = toy.struct_constant [
    dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>,
    dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
  ] : !toy.struct<tensor<*xf64>, tensor<*xf64>>
  %1 = toy.generic_call @multiply_transpose(%0) : (!toy.struct<...>) -> tensor<*xf64>
  toy.print %1 : tensor<*xf64>
  toy.return
}
```

Reading this:
- `!toy.struct<tensor<*xf64>, tensor<*xf64>>` — our custom struct type with two tensor fields
- `toy.struct_access %arg0[0]` — accessing the first field (index 0)
- `toy.struct_constant [...]` — a constant struct with literal tensor values

---

## Key Source Files

| File | What it does |
|------|-------------|
| `include/toy/Ops.td` | Declares toy.struct_constant and toy.struct_access ops |
| `include/toy/Types.td` | Declares the StructType custom type |
| `mlir/Dialect.cpp` | Implements StructType print/parse, registers it |
| `mlir/MLIRGen.cpp` | Handles struct AST nodes, emits struct ops |
| `include/toy/Parser.h` | Parses struct declarations and field access |
| `mlir/LowerToAffineLoops.cpp` | Lowering patterns for struct ops |

---

## The Command Run

```cmd
toyc-ch7.exe struct-codegen.toy -emit=mlir
```

What this does:
1. Lexer + Parser → AST (now handles struct syntax)
2. MLIRGen → MLIR with `!toy.struct` type and struct ops
3. `-emit=mlir` prints the result

See `output.txt` for the full MLIR output with struct types.

---

## The Big Lesson of Chapter 7

MLIR's type system is **open and extensible**. You are not limited to tensors, integers, and floats. You can define any type that makes sense for your language or domain, and it integrates naturally with:
- The existing dialect infrastructure
- Shape inference interfaces
- The lowering pipeline
- The print/parse system

This is why MLIR is used for domain-specific compilers — a hardware company can define types specific to their chip, an AI framework can define types specific to neural network layers, and they all work within the same compiler framework.
