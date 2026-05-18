# Chapter 7 - Adding a Custom Struct Type

**Tutorial:** https://mlir.llvm.org/docs/Tutorials/Toy/Ch-7/

## What This Chapter Does
Extends the Toy language with a custom `struct` composite type, demonstrating how to add custom types to MLIR.

## Command Used

```bash
toyc-ch7.exe struct-codegen.toy -emit=mlir
```

## Output
See `output.txt` for the MLIR output showing `!toy.struct` types and `toy.struct_access` operations.

## New Language Feature
```toy
struct MyStruct {
  var a<2, 3>;
  var b<2, 3>;
}
```

## New Operations
- `toy.struct_constant` — a constant struct literal
- `toy.struct_access` — access a field by index

## Key Concepts
- **TypeDef in ODS**: Declarative custom type definition
- **ParametricType**: Type with runtime parameters (the field list)
- Custom print/parse via `assemblyFormat`