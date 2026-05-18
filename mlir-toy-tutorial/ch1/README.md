# Chapter 1 — Toy Language and AST

**Tutorial:** https://mlir.llvm.org/docs/Tutorials/Toy/Ch-1/

---

## What This Chapter Is About

This is the starting point of the compiler. Before any MLIR is involved, we need to **read the Toy source code and understand its structure**. That is what this chapter builds:

1. A **Lexer** — reads the raw text character by character and breaks it into tokens
2. A **Parser** — takes those tokens and builds a tree structure called an AST
3. An **AST (Abstract Syntax Tree)** — a tree that represents the meaning of the program

There is zero MLIR in this chapter. This is pure compiler frontend work.

---

## What is a Lexer?

A lexer (also called a tokenizer or scanner) reads your source file and groups characters into meaningful units called **tokens**.

For example, this Toy source line:
```
var a<2, 3> = [[1, 2, 3], [4, 5, 6]];
```

Gets broken into tokens like:
```
KEYWORD(var)  IDENT(a)  <  NUMBER(2)  ,  NUMBER(3)  >  =  [  [  NUMBER(1)  ...
```

The lexer does not understand meaning — it just recognizes patterns like "this is a number", "this is an identifier", "this is a keyword".

---

## What is a Parser?

The parser takes the stream of tokens from the lexer and builds a **tree** that represents the structure of the program. The Toy parser is a hand-written **recursive descent parser** — meaning it has one function per grammar rule that calls other functions recursively.

For example, when it sees `def multiply_transpose(a, b) { ... }`, it knows this is a function definition and builds a `FunctionAST` node containing:
- The function name
- The parameter names
- A block of statements (the body)

---

## What is an AST?

The AST (Abstract Syntax Tree) is a tree where each node represents one construct of the language. It is "abstract" because it drops unimportant details (like parentheses and semicolons) and keeps only the structure.

For the Toy program:
```toy
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}
```

The AST looks like:
```
FunctionAST: multiply_transpose
  params: [a, b]
  body:
    ReturnStmt
      BinaryExprAST: *
        CallExprAST: transpose
          VariableExprAST: a
        CallExprAST: transpose
          VariableExprAST: b
```

Every later chapter works on top of this AST — Chapter 2 walks it to generate MLIR.

---

## Key Source Files

| File | What it does |
|------|-------------|
| `include/toy/Lexer.h` | Reads source text, produces tokens one by one |
| `include/toy/Parser.h` | Consumes tokens, builds typed AST node objects |
| `include/toy/AST.h` | Defines all AST node types (FunctionAST, BinaryExprAST, etc.) |
| `toyc.cpp` | Main driver: calls parser, then dumps the AST |

---

## The Command Run

```cmd
toyc-ch1.exe ast.toy -emit=ast
```

What this does step by step:
1. `toyc-ch1.exe` loads the Toy compiler binary for Chapter 1
2. `ast.toy` is the input Toy source file
3. The Lexer reads `ast.toy` and produces tokens
4. The Parser consumes the tokens and builds the AST tree
5. `-emit=ast` tells the compiler: "stop here and print the AST"
6. The AST is printed to the terminal

See `output.txt` for the full AST dump from running this command.

---

## What the Output Means

In `output.txt` you will see lines like:

```
Module:
  Function
    Proto 'multiply_transpose' @ast.toy:4:1
    Params: [a, b]
    Block {
      Return
        BinOp: * @ast.toy:5:25
          Call 'transpose' [
            var: a
          ]
          Call 'transpose' [
            var: b
          ]
    }
```

Reading this:
- `Proto 'multiply_transpose' @ast.toy:4:1` — a function named `multiply_transpose` defined at line 4, column 1
- `BinOp: *` — a multiplication operation
- `Call 'transpose'` — a call to the transpose function
- `var: a` — the variable `a` passed as argument
- `@ast.toy:5:25` — source location for error messages

---

## Why This Matters

The AST is the foundation of everything that follows. If the AST is wrong, every subsequent chapter will produce wrong output. Getting the lexer and parser right means the rest of the compiler has a reliable input to work from.

Also, **storing source locations** (file, line, column) in every AST node is critical for producing useful error messages later. MLIR carries this location information all the way through the pipeline.
