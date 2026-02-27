# lp-common

Shared utilities for the Light Player compiler stack (cranelift, regalloc, RISC-V tools).

## Crates

### lp-collection

Embedded/low-memory friendly collections that allocate in small chunks to reduce OOM risk from heap fragmentation:

- **ChunkedVec** - Vector backed by multiple smaller allocations instead of one large contiguous block
- **ChunkedHashMap** - Hash map with fixed 64 buckets, each backed by a ChunkedVec

Used by regalloc2 (fastalloc) and cranelift-codegen (VCode).

## Versioning

Releases are automated via [release-plz](https://release-plz.dev/) using conventional commits:
- `fix:` -> patch (1.0.0 -> 1.0.1)
- `feat:` -> minor (1.0.0 -> 1.1.0)
- `feat!:` or `BREAKING CHANGE` -> major (1.0.0 -> 2.0.0)

## Usage

Add as a git dependency:

```toml
lp-collection = { git = "https://github.com/light-player/lp-common", branch = "main" }
# or pin to a release:
lp-collection = { git = "https://github.com/light-player/lp-common", rev = "v1.0.0" }
```
