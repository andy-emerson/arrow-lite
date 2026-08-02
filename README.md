# arrow-lite

**Aligned, typed, strided numeric buffers with Arrow-compatible layout and a C
Data Interface — and nothing else.**

> **Status:** Design phase. Nothing is implemented. This README describes what
> `arrow-lite` is intended to be; [`DESIGN.md`](DESIGN.md) carries the
> invariants, settled decisions, and the open questions that are still open.

`arrow-lite` is the memory substrate two other projects are built on:
[MatLua](https://github.com/andy-emerson/MatLua) (dense numeric arrays and
linear algebra for Lua) and `df-lite` (SQL planning and execution over ordered
numeric data), with [TallyDB](https://github.com/andy-emerson/TallyDB) sitting
on top of both. It exists because those three need to hand each other buffers
without copying, and the only way to guarantee that is for all of them to
allocate from the same place with the same rules.

It implements a strict subset of the [Apache Arrow columnar
format](https://arrow.apache.org/docs/format/Columnar.html) — three data types
instead of forty — and implements that subset **byte-identically**. A consumer
that only ever sees `f64`, `i64`, and dictionary-encoded key columns cannot
tell it is not talking to arrow-rs.

## What it is

- **Three physical types.** `f64`, `i64`, and `u32` dictionary codes. Each is
  fixed-width and constant-stride. There is no fourth.
- **One allocator.** 64-byte aligned, used by every crate in the stack, so a
  buffer produced by one is indistinguishable from a buffer produced by
  another.
- **One buffer type.** `Buf` — pointer, length, stride, dtype, alignment,
  provenance. Owned and borrowed are the same type with different provenance,
  so a function that takes a `Buf` does not care who allocated it.
- **Validity bitmaps**, to Arrow's spec: one bit per element, LSB-first, with
  logical offsets rather than re-packing on slice.
- **Dictionary encoding** for key columns, including the cross-segment
  remapping that a segmented store needs.
- **The Arrow C Data Interface** — `ArrowArray`, `ArrowSchema`, and
  `ArrowArrayStream` — so that PyArrow, DuckDB, Polars, and anything else
  Arrow-aware can read our buffers without either side linking the other's
  stack.

## Quick feel

```rust
use arrow_lite::{Buf, DType, Validity};

// Allocate. 64-byte aligned, always.
let mut xs = Buf::zeros(DType::F64, 1024)?;
xs.as_mut_slice::<f64>()?[0] = 3.5;

// Borrow a strided view. No copy, no allocation.
let every_fourth = xs.view(0..1024, 4)?;
assert_eq!(every_fourth.len(), 256);

// Wrap memory somebody else owns. Same type, different provenance.
let borrowed = unsafe { Buf::from_raw(ptr, len, DType::I64) };

// Hand it to anything Arrow-aware without either side linking the other.
let (array, schema) = xs.to_arrow()?;   // ArrowArray + ArrowSchema
```

A function that takes `&Buf` cannot tell whether the memory was allocated
here, allocated by a storage engine, or borrowed from a foreign process. That
is the entire point.

## What you can rely on

- **64-byte alignment** on every owned allocation. Safe to hand to SIMD
  kernels without a preamble.
- **`len` and `stride` count elements, never bytes.**
- **Byte-identical Arrow layout** for every type implemented. Verified against
  arrow-rs and PyArrow in CI, in both directions, at every slice offset.
- **Null slots are never read as data.** A null's underlying bytes are
  unspecified and this crate will not interpret them.
- **No panic crosses the FFI boundary**, and no allocation happens on a path
  documented as allocation-free.

## What it is not

- **Not a compute library.** No kernels, no arithmetic, no aggregation. It
  hands you a pointer and a length; what you do with them is your business.
  MatLua does math.
- **Not a table library.** No schemas, no field names, no `RecordBatch`, no
  `Table`. Those are tabular concepts and they belong above an array
  abstraction, not below it. `df-lite` does tables.
- **Not an array library.** No rank, no shape, no broadcasting, no reshape. A
  `Buf` is one-dimensional. MatLua layers n-D semantics on top.
- **Not an implementation of Arrow.** No IPC, no Feather, no Parquet, no
  strings, no nested types, no timestamps as distinct physical types, no
  compute. If you want Arrow, use
  [arrow-rs](https://github.com/apache/arrow-rs) — it is complete, mature, and
  correct, and this crate is not trying to replace it.

## Why not just use arrow-rs

Because the projects downstream of this crate are betting on being small enough
to embed and simple enough to reason about, and arrow-rs is neither small nor
simple — by design, and correctly, because it implements the whole format.
Three dtypes with no compute kernels, no IPC, and no Parquet is a different
artifact with a different cost profile: fast to compile, small to link, and
short enough that the unsafe core can be read end to end in an afternoon.

The subset is also the point. TallyDB's design turns on every column being
fixed-width and constant-stride so that a math library can read it directly.
A buffer layer that admits variable-width types has already given that away.

**arrow-rs is our oracle, not our dependency.** It appears in
`[dev-dependencies]` and nowhere else. Every layout this crate produces is
round-trip tested against both arrow-rs and PyArrow in CI — two independent
implementations, because agreement between two is evidence about the
specification and agreement with one is evidence about that library's quirks.

## The one place we diverge from Arrow

Arrow arrays are contiguous with a logical offset; they have no concept of a
stride. `Buf` carries one, because the consumers need non-contiguous column
views and the alternative is compacting before every call.

**The rule you need to know:** a strided `Buf` is fully valid inside this
stack, and `to_arrow` requires unit stride — compacting if the buffer is
strided. So export may copy where the rest of the crate does not. Nothing
outside the stack ever observes a non-Arrow layout.

`DESIGN.md`, *Strides*, carries the reasoning and the rejected alternative.

## Who depends on what

```
arrow-lite  ──►  (nothing)
   ▲    ▲
   │    └──────  MatLua  ──►  faer
   │                  ▲
   └──  df-lite  ──►  sqlparser
           ▲          ▲
           └─ TallyDB ┘  ──►  MatLua
```

Everything points down. Nothing points sideways. arrow-rs appears only in dev
dependencies, at every level.

## Stability

Design phase. The API will change. Until a `0.1.0` cut, treat anything here as
provisional — including the layout, though layout changes will be visible as
diffs against committed golden bytes rather than silent.

This is deliberately unambitious code: no research, no novel algorithms, every
piece specified elsewhere. The work is to get it right once and then stop
touching it.

## How we work

This repository follows the working agreement in
[`AGENTS.md`](https://github.com/andy-emerson/working-agreement).

- **Durable documents:** this README and [`DESIGN.md`](DESIGN.md).
- **Living status:** GitHub Issues. Open decisions carry the `decision` label.
  Settled decisions — including rejected alternatives and their reopen
  triggers — live in `DESIGN.md`, not in open issues.
- **Checks:** fmt, clippy, build, tests including doctests, rustdoc with
  warnings as errors, Miri over the unsafe core, ASan and UBSan over the FFI
  surface, and differential round-trip tests against arrow-rs and PyArrow.
- **Audience:** documentation for a reader with a BS in applied mathematics and
  a CS minor; code for the CS-minor side.

## License

MIT.
