# Chapter 6 — Lowering to LLVM IR and Code Generation

**Tutorial:** https://mlir.llvm.org/docs/Tutorials/Toy/Ch-6/

---

## What This Chapter Is About

This is the final stage of the compiler pipeline. We lower everything — including the remaining `toy.print` op — all the way down to **LLVM IR** (LLVM's Intermediate Representation). Then we use **LLVM's JIT compiler** to actually execute the program and see real output printed to the screen.

This is the chapter where the Toy program goes from being just IR text to actually **running as a real program**.

---

## The Complete Lowering Pipeline

Here is the full journey from Chapter 5 to execution:

```
After Chapter 5:
  affine.for loops + memref buffers + arith ops + toy.print (still remaining)
        │
        ▼  LowerToLLVM pass
  llvm dialect ops (llvm.func, llvm.call, llvm.store, llvm.br, etc.)
        │
        ▼  translateModuleToLLVMIR()
  LLVM IR (textual .ll format — what you see with -emit=llvm)
        │
        ▼  LLVM LLJIT
  Native machine code
        │
        ▼
  Program executes, output printed to screen!
```

---

## What Gets Lowered in Chapter 6?

### toy.print → printf calls

`toy.print` takes a memref tensor and prints every element. It gets lowered to:
1. A global format string constant: `"%f \n"` stored in the binary
2. Nested loops over all tensor dimensions
3. A `printf` call for each element
4. A newline `printf` call after each row

The result in LLVM IR looks like:
```llvm
@frmt_spec = internal constant [4 x i8] c"%f \00"
@nl = internal constant [2 x i8] c"\0A\00"

; For each element:
%val = load double, ptr %element_ptr
call i32 @printf(ptr @frmt_spec, double %val)
; After each row:
call i32 @printf(ptr @nl)
```

### Everything else → llvm dialect

All remaining `affine`, `memref`, `arith`, and `func` ops are lowered using MLIR's built-in conversion passes:
- `ConvertAffineToStandard` — affine.for → scf.for (structured control flow)
- `ConvertSCFToControlFlow` — scf.for → cf.br (raw branches)  
- `ConvertArithToLLVM` — arith.mulf → llvm.fmul
- `ConvertFuncToLLVM` — func.func → llvm.func
- `FinalizeMemRefToLLVM` — memref.alloc → llvm.call @malloc

---

## What is LLVM IR?

LLVM IR is the low-level intermediate representation used by the LLVM compiler infrastructure. It is essentially a typed assembly language that is:
- **Platform independent** (no mention of registers or specific instructions)
- **Strongly typed** (every value has a type like `i64`, `double`, `ptr`)
- **In SSA form** (every value assigned exactly once)

After this IR is generated, LLVM's backend translates it to actual machine instructions for your CPU (x86, ARM, etc.).

---

## What is LLJIT?

LLJIT is LLVM's **Just-In-Time compiler**. Instead of writing the program to a `.exe` file, JIT:
1. Takes the LLVM IR in memory
2. Compiles it to machine code in memory
3. Finds the `main` function
4. Jumps to it and executes immediately

This is how many interpreted languages work under the hood (Julia, JavaScript V8, PyPy, etc.).

With `-emit=jit`, the Toy compiler actually runs your program and you see the computed numbers printed.

---

## What the -emit=llvm Output Shows

In `output.txt` (from `-emit=llvm`), you can see real LLVM IR:

```llvm
; Allocate memory for the result tensor (3x2 doubles = 48 bytes)
%1 = call ptr @malloc(i64 48)

; Store the constant values 1.0, 2.0, 3.0... into memory
store double 1.000000e+00, ptr %26, align 8

; The transpose loop (nested br/phi instead of affine.for)
br label %37
37:
  %38 = phi i64 [ 0, %0 ], [ %56, %55 ]
  %39 = icmp slt i64 %38, 3
  br i1 %39, label %40, label %57

; The multiply loop
%71 = fmul double %70, %70

; The print loop
%93 = call i32 (ptr, ...) @printf(ptr @frmt_spec, double %92)

; Free memory when done
call void @free(ptr %99)
ret void
```

This is what your Toy program looks like in the eyes of LLVM — pure memory operations, loops as branch instructions, and system calls to `malloc`, `free`, and `printf`.

---

## Key Source Files

| File | What it does |
|------|-------------|
| `mlir/LowerToLLVM.cpp` | PrintOp → printf lowering pattern + LowerToLLVMPass |
| `mlir/LowerToAffineLoops.cpp` | Same as Ch5, still needed |
| `toyc.cpp` | Adds `-emit=llvm` and `-emit=jit` options + JIT execution code |

---

## The Command Run

```cmd
toyc-ch6.exe codegen.toy -emit=llvm
```

What this does:
1. Lexer + Parser → AST
2. MLIRGen → toy dialect MLIR
3. LowerToAffineLoops pass → affine + memref + arith (same as Ch5)
4. LowerToLLVM pass → llvm dialect (including toy.print → printf)
5. `translateModuleToLLVMIR()` → LLVM IR
6. `-emit=llvm` prints the LLVM IR text

See `output.txt` for the full LLVM IR.

---

## The Full Journey Complete

Starting from a simple Toy source file, after 6 chapters we have a working compiler that:
- Reads and parses the Toy language
- Generates MLIR in the toy dialect
- Optimizes the IR using pattern rewriting
- Inlines functions and infers tensor shapes
- Lowers to affine loops and memory operations
- Generates LLVM IR with malloc, printf, and actual loops
- Can JIT-compile and execute the program natively

This is a complete, working compiler pipeline built using MLIR.
