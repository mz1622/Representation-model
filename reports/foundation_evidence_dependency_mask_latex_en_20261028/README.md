# Dependency-Mask Foundation Evidence: English LaTeX Release

This directory contains the English-only LaTeX version of the revised dependency-mask evidence report.

## Contents

- `main.tex`: complete LaTeX source.
- `figures/`: the eight aggregate figures referenced by `main.tex`.
- `tables/`: the 18 aggregate CSV files needed to verify the reported values.

No raw database, Taiwan source file, row-level prediction, derived profile matrix, or model checkpoint is included.

## Build

Compile `main.tex` from this directory so that relative `figures/` paths resolve correctly. The source was successfully compiled and checked with the Codex built-in LaTeX toolchain before commit.
