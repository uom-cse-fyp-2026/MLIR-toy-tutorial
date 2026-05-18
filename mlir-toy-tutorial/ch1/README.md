# Chapter 1 - Toy Language & AST

**Tutorial:** https://mlir.llvm.org/docs/Tutorials/Toy/Ch-1/

## What This Chapter Does
Introduces the Toy language and builds a Lexer, Parser, and AST (Abstract Syntax Tree). No MLIR yet — this is purely the compiler frontend.

## Command Used

```bash
toyc-ch1.exe ast.toy -emit=ast
```


## Output
See `output.txt` for the full AST dump.

## Key Concepts
- **Lexer**: Scans source text and produces tokens
- **Parser**: Recursive-descent parser that builds AST nodes
- **AST**: Tree where each node is a language construct (function, variable, binary op, etc.)