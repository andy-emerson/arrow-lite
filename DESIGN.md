# arrow-lite — Design

The design companion to [`README.md`](README.md). The README says what
`arrow-lite` is; this document says why, records what has been settled
(including what was rejected and what would reopen it), and lists what has not
been settled yet.

**Status: design phase.** Nothing here has been built. Sections marked
**OPEN** are genuinely undecided and are the ones to argue about.

---

## 1. Philosophy

### 1.1 The crate exists to make one claim true

The claim is: *a buffer allocated by TallyDB's storage layer can be read by
MatLua's linear algebra kernels with no conversion, no copy, and no
reinterpretation.*

That claim is easy to state and easy to lose. It is lost the moment two crates
in the stack have their own buffer types, because then the seam between them
needs a conversion, and a conversion in a per-window loop is the entire
performance budget. TallyDB measured a general LAPACK solver spending 2.3µs of
a 2.5µs window on allocation. There is no room.

So `arrow-lite` is not primarily a library. It is a **shared contract with an
implementation attached**. Its value is that everything upstream agrees on it.

### 1.2 Minimal means minimal

Every type admitted here multiplies work upstream: a kernel in MatLua, a
codec in TallyDB, an operator path in `df-lite`, and a differential test
against two reference implementations for each. Three types is not asceticism;
it is the number that the workload actually requires, and adding a fourth
should feel expensive because it is.

The test for admitting a physical type is narrow: **does it require an
addressing scheme none of the existing types provide?** Not "would it be
convenient," not "does Arrow have it." Addressing.

Anything wanting new *meaning* is a logical annotation (§4.4). Anything wanting
new *space efficiency* is a codec, and codecs live in TallyDB, not here.

### 1.3 The specification is the contract; arrow-rs is an oracle

Conformance target is the [Arrow columnar
format](https://arrow.apache.org/docs/format/Columnar.html) and the [C Data
Interface](https://arrow.apache.org/docs/format/CDataInterface.html). arrow-rs
is one implementation of those documents, not the definition of them.

Where arrow-rs does something the specification does not require, we follow the
specification. Where the specification is ambiguous, we follow arrow-rs *and
PyArrow together*, and record the ambiguity here.

arrow-rs and PyArrow are `[dev-dependencies]` and CI harnesses. Neither is ever
a runtime dependency. If that changes, this crate has failed at its purpose.

---

## 2. Invariants

These are load-bearing. Violating any one of them breaks something upstream.

1. **Three physical types: `f64`, `i64`, `u32`.** Fixed-width, constant-stride,
   byte-addressable. No fourth type without a change to this document.
2. **64-byte alignment on every allocation.** No exceptions, no smaller
   fallback. This is what makes buffers SIMD-safe and interchangeable.
3. **One allocator.** Every buffer in the stack comes from `arrow-lite`'s
   allocator. A `Vec<f64>` promoted into a `Buf` is not a `Buf`.
4. **Owned and borrowed are the same type.** A function taking a `&Buf` does
   not know or care who owns the memory.
5. **No compute.** Not one arithmetic kernel. The moment a `sum` lands here,
   the boundary with MatLua stops being obvious and starts being negotiable.
6. **No schema.** No field names, no types-with-names, no tables. `df-lite`
   owns tabular concepts.
7. **No rank.** A `Buf` is one-dimensional. MatLua owns shape.
8. **Byte-identical to Arrow for every type we implement.** Subset in breadth,
   never in fidelity.

---

## 3. The type system

### 3.1 Three physical types, three roles

The types are not three arbitrary numeric choices. They correspond to three
algebraic roles that the ordered-numeric workload requires, and each is the
minimal type that satisfies its role.

| Type  | Role | Requirement it satisfies | Why not another type |
|-------|------|--------------------------|----------------------|
| `i64` | Ordering key, exact measures | Total order, monotone, exact addition and subtraction | `f64` is exact only to 2⁵³; nanosecond epochs (~1.7×10¹⁸) passed that in the 1970s. Delta encoding on floats is lossy and non-monotone in the low bits. |
| `f64` | Continuous measures | Real arithmetic, direct handoff to a dense solver | `f32` halves the kernel surface's precision and doubles its breadth for a memory win better obtained as a codec. |
| `u32` | Dictionary codes for keys | Equality, hashing, dense grouping | `u8`/`u16` overflow is a correctness cliff under unbounded-cardinality ingest, and mixed widths complicate cross-segment remapping. `u64` wastes half of every key column. |

`i64` and `f64` are consumed by arithmetic. `u32` never is — codes are
compared, hashed, and grouped, never summed. That asymmetry is why `u32` is
admitted despite being narrower: it is only ever asked to satisfy the weaker
requirement.

### 3.2 What is excluded, and why

**`bool`.** Arrow's Boolean is bit-packed: one bit per element. Element *i* has
no byte address; reading it is a shift and a mask, not a load. That is a third
addressing scheme, and every kernel that assumes `base + i * stride` would need
a second implementation — which is precisely the per-value branching the stack
exists to remove. A boolean is expressible as `i64` 0/1, which costs 64× in
memory and recovers nearly all of it under RLE at the codec layer, and which
has the compensating property that it sums to a count, averages to a rate, and
drops into a design matrix as a dummy variable. A bit-packed column can do none
of those without unpacking.

*Considered and rejected: `bool` as a key type.* Reclassifying it from numeric
to key changes the role, not the layout — the bit-packing problem is
unaffected. It additionally incurs dictionary bookkeeping at cardinality 2,
where dictionary encoding has no benefit to offer, and forfeits the arithmetic
above. Rejected on three counts.

**Strings and other variable-width types.** They require an offsets buffer, so
element *i* costs a load to locate. That defeats constant-stride scanning,
which is the property everything upstream is built on. Non-negotiable.

**`f32`.** Doubles the numeric kernel surface for a memory win available as a
codec (store narrowed, decode to `f64` on read). *Reopen trigger:* if TallyDB's
scans prove L1/L2-bound rather than IO-bound, the codec approach keeps the disk
and page-cache win but loses the cache-residency win, and `f32` as a stored
type becomes worth re-examining.

**Complex.** Would need an interleaved-pair layout — a genuinely new addressing
scheme, so it clears the §1.2 bar in principle. But Arrow itself has no complex
type (proposals date to 2017 and remain unadopted), nothing upstream wants one,
and adopting a non-standard layout would break the byte-identity invariant.
*Reopen trigger:* Arrow adopting a canonical complex type, plus a concrete
upstream need.

**`Decimal128`.** Sixteen bytes, no invertibility advantage over scaled `i64`,
needs its own arithmetic kernels everywhere. Money in minor units as `i64`
covers the ledger case.

**`f16`, unsigned types other than `u32`, `i32` as a measure.** Each either
duplicates kernels or forces a widening pass with no compensating win.

---

## 4. The `Buf` contract

### 4.1 Shape of the type

```rust
pub struct Buf {
    ptr:    NonNull<u8>,
    len:    usize,        // element count, not bytes
    stride: usize,        // element stride; 1 is contiguous
    dtype:  DType,        // F64 | I64 | U32
    owner:  Provenance,   // owned allocation, or a borrow with a lifetime
}
```

The exact representation is an implementation detail and will change. What is
contractual:

- `len` counts **elements**, never bytes.
- `stride` counts **elements**, never bytes. `stride == 1` is contiguous.
- Alignment is 64 bytes for owned allocations. Borrowed views inherit whatever
  the underlying allocation had, which under invariant 3 is also 64.
- Owned and borrowed are the same type. Provenance is internal.

### 4.2 Strides — SETTLED, and the one deliberate divergence

**Decision: `Buf` carries a stride. C Data Interface export requires unit
stride and compacts if the buffer is strided.**

Arrow has no strides. Arrow arrays are contiguous with a logical offset. So
this is a divergence from the format, and it is deliberate:

- MatLua's array core needs strided views (`slice`, `col`, `transpose`,
  `broadcast_to`). Without strides in `Buf`, MatLua wraps `Buf` in its own view
  type and we are back to two buffer species at the seam.
- TallyDB's column slices are not always contiguous — segment boundaries and
  tombstone resolution both produce gapped access. Requiring contiguity means
  compacting before every call, which is the copy this stack exists to avoid.

*Rejected alternative: strides live in MatLua, `arrow-lite` stays purely
contiguous.* Cleaner conformance, but it reintroduces the seam and makes
`arrow-lite` a type MatLua wraps rather than a type MatLua uses. The whole
argument for a shared foundation crate is that both consumers use the same
type.

*Containment rule:* the divergence is visible only at the export boundary. A
strided `Buf` is fully valid inside the stack; `to_arrow` demands unit stride
and compacts otherwise, so nothing outside the stack ever observes a
non-Arrow layout. This costs a copy on export, which is acceptable because
export is the cold path by construction.

*Reopen trigger:* if compaction-on-export turns out to be hot in practice — a
result set that is strided all the way out — reconsider whether the query layer
should be materializing contiguously before export instead.

### 4.3 Validity — SETTLED, with a known asymmetry

**Decision: Arrow's bitmap. One bit per element, LSB-first within each byte,
`null_count` cached, slicing via logical offset rather than re-packing.**

This is the format's layout and byte-identity requires it.

**Known asymmetry, recorded so it is not rediscovered later.** TallyDB's
Lua-facing vector (`tallydb.vector`) uses **one validity byte per element**,
not a bitmap. That is a defensible choice on its side — byte-addressable, no
shift-and-mask in scripting-tier code — but it means the two representations do
not line up, and a crossing between the Lua vector and an `arrow-lite` buffer
costs a bitmap↔byte-mask expansion in both directions.

This is not a bug. It is two boundaries with two different cost profiles: the
Lua boundary is crossed once per query result, the buffer boundary is crossed
once per window. Optimizing the frequent one is correct. But the conversion
should be written once, here or in TallyDB, and not reinvented per call site.

**OPEN:** where the bitmap↔byte-mask conversion lives. Candidates:
`arrow-lite` (it owns both representations conceptually), or TallyDB (it owns
the Lua vector). Leaning `arrow-lite`, since MatLua may eventually want it too.

**Non-goal:** `arrow-lite` never interprets the contents of a null slot. The
bytes under a null are unspecified and must not be read as data. This is the
requirement TallyDB's letter §3.4 names, and it is upheld here as an invariant
rather than a behaviour.

### 4.4 Logical annotations — SETTLED in principle, OPEN in placement

An Arrow `Timestamp` is physically an `int64` plus a metadata string naming
unit and timezone. So are `Date64`, `Time64`, and `Duration`. Adding them
requires **no new layout and no new kernel** — only a tag that travels with the
buffer and is emitted on export.

The same trick covers `bool` (a tag on `i64` 0/1) and `categorical` (a tag on a
dictionary key column).

This is cheap and it buys real things: PyArrow sees a timezone-aware timestamp
column instead of a bare integer, and `df-lite` can implement `date_trunc`,
`extract`, and `date_bin` as `i64 → i64` functions that know what unit they are
operating on.

**OPEN:** whether the annotation lives on `Buf` or in `df-lite`'s schema. On
`Buf` means `arrow-lite` carries a concept MatLua will never read. In the
schema means the CDI export path needs the schema to emit the right format
string, which it arguably needs anyway. Leaning schema, with `arrow-lite`'s
export API taking the annotation as a parameter rather than storing it.

### 4.5 Dictionary encoding — SETTLED, lives here

**Decision: dictionary encoding is in `arrow-lite`.**

Argued the other way first: MatLua has no use for dictionaries, and the "only
what both consumers need" rule would put them in `df-lite`.

But TallyDB's **on-disk format** stores key columns dictionary-encoded, and
storage sits below query. If dictionaries lived in `df-lite`, TallyDB's storage
layer would depend on a query crate to read its own bytes. That is backwards
enough to override the rule, and the rule takes an explicit exception for
format-level concerns.

Contents: codes buffer (`u32`) plus values, a builder that interns during
ingest, and **cross-segment remapping** — merging two segments' dictionaries
and rewriting codes — which is the subtle part and the reason this is 400–600
lines rather than 100.

### 4.6 C Data Interface — SETTLED

`ArrowArray`, `ArrowSchema`, `ArrowArrayStream`, to spec.

**Release-callback discipline**, stated explicitly because mis-handling it
produces a leak or a double free in someone else's production process rather
than in a test:

- Whoever produces a struct supplies its `release` callback.
- The consumer calls `release` exactly once.
- A moved-from struct has its `release` field nulled.
- `release` must be safe to call on a partially-initialized struct.
- No Rust value requiring `Drop` may be live across a call that can `longjmp`.
  (This matters upstream in MatLua's Lua bindings; it is stated here because
  the release callbacks are the FFI surface this crate owns.)

**No panic crosses the boundary.** Ever. Errors are `Result` up to the edge and
converted there.

This is the third of the three careful spots, and it gets ASan and UBSan with
adversarial release ordering in CI, not just unit tests.

---

## 5. Errors

One error type, shared by all three consumers, defined here. Not because
`arrow-lite` has interesting errors — it has about five — but because three
crates each defining their own means conversion impls at every seam.

Kinds: allocation failure, dtype mismatch, length mismatch, stride requirement
violated (export path), null present where not permitted, CDI protocol
violation.

---

## 6. Open questions

Everything in this section is genuinely undecided.

**6.1 Builders: here or in TallyDB?** Appending rows one at a time into a
growing typed buffer is TallyDB's write-buffer path, and it is the only
consumer. But it is also pure buffer manipulation with no storage semantics.
300–500 lines either way. *Leaning: here, because MatLua's `zeros`/`arange`
constructors want the same growth machinery.*

**6.2 `no_std` + `alloc`?** Would matter for a WASM build, which TallyDB lists
as a future direction. Costs discipline now, buys optionality later. *Leaning:
yes, because the cost is low if adopted from the start and high if retrofitted.*

**6.3 Where does bitmap↔byte-mask conversion live?** See §4.3.

**6.4 Where do logical annotations live?** See §4.4.

**6.5 Does `Buf` need a "chunked" concept?** TallyDB returns one batch per
segment, so a logical column spans several buffers. Today that is the caller's
problem — a `&[Buf]`. If MatLua's gather path ends up needing to iterate
across chunks, a first-class chunked view might belong here. *Leaning: no. Keep
it a caller concern until something forces it.*

**6.6 Alignment on borrowed views.** If a borrow starts at element 7 of a
64-byte-aligned buffer, the view's base pointer is not 64-byte aligned. Does
the contract require realignment, or does it only promise alignment for owned
allocations? *This has SIMD consequences upstream and should be settled before
any kernel is written against it.*

---

## 6a. Risk and size

Expected source: roughly 3,500 lines, plausible range 2,500–5,000 depending on
§6.1. Expected tests: 1.5–2.5× that. If the tests are not larger than the
source, the crate is not doing its job.

Rough allocation: `Buf` and views 500–800; allocator 150–250; dtypes and typed
access 200–350; validity 300–450; dictionary 400–600; CDI 600–900; stream
200–300; errors 100–150; builders 300–500.

**Nothing here is research.** Every piece has a published specification or a
well-known correct implementation. The risk is *defect*, not *discovery*, which
means the schedule is estimable and the review burden is front-loaded.

Three parts carry disproportionate risk, all with the property that being
almost-right produces a bug in someone else's production process rather than a
failing test:

1. **The CDI release protocol.** Ownership across FFI with a callback. Failure
   modes are double-free and leak, both non-local. Mitigation: ASan and UBSan
   with adversarial release ordering, not unit tests.
2. **Bitmap slicing at non-byte offsets.** Slicing at bit 3 shifts everything
   downstream. Fully specified, easy to get subtly wrong. Mitigation: property
   tests at every offset 0–63 against a naive implementation.
3. **Stride × validity interaction.** A strided view over a bitmap has no
   analogue in Arrow, so there is no oracle for it. This is the only part of
   the crate where we are on our own, and it is the reason §6.6 must be settled
   before any kernel is written.

## 7. Test plan

The crate is unsafe-heavy and small, which is the right shape for exhaustive
testing.

- **Differential round-trip against arrow-rs and PyArrow.** Every dtype, every
  nullability combination, every slice offset including non-byte-aligned ones.
  Both directions. Two independent oracles, not one.
- **Miri** over the entire unsafe core.
- **ASan and UBSan** over the CDI surface, including adversarial release
  ordering: release-before-consume, double release, release of a moved-from
  struct, release of a partially-initialized struct.
- **Property tests** on bitmap slicing: slice at every bit offset from 0 to 63,
  confirm popcount and per-element reads agree with a naive implementation.
- **Property tests** on dictionary remapping: merge random dictionaries,
  confirm every code maps to the same value before and after.
- **Doctests as the preferred executable evidence**, per the working agreement.
- **Golden byte tests.** Committed reference bytes for each dtype's layout, so
  that a layout change is a deliberate act with a visible diff rather than an
  accident.

---

## 8. Build order

1. `Buf`, dtypes, allocator, alignment. Miri from the first commit.
2. Validity bitmaps, including slicing at non-byte offsets.
3. C Data Interface export, then import. Differential tests against both
   oracles as soon as export exists — this is the gate on everything else.
4. `ArrowArrayStream`.
5. Dictionary encoding and cross-segment remapping.
6. Builders (pending §6.1).

Steps 1–4 are enough for MatLua to start. Step 5 is what TallyDB's storage
layer waits on.

---

## 9. Who we write for

Documentation for a reader with a BS in applied mathematics and a CS minor.
Code for the CS-minor side. Where the two diverge — and in an unsafe buffer
library they diverge often — the comment explains the invariant and the code
enforces it.
