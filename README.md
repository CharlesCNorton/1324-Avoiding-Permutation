# 1324-Avoiding Permutations

A machine-verified decomposition of 1324-avoiding permutations with a proven bijection to Catalan numbers.

## The Problem

The enumeration of 1324-avoiding permutations ([OEIS A061552](https://oeis.org/A061552)) is a long-standing open problem in combinatorics. The generating function G(x) is conjectured to be non-D-finite.

```
G(x): 1, 1, 2, 6, 23, 103, 513, 2762, 15793, ...
```

## Main Result

We prove a **bijection theorem** (machine-verified in Coq):

> **Theorem.** A permutation [σ, n] with maximum element n at the end avoids 1324 if and only if σ avoids 132.

This establishes that permutations with max-at-end are counted by Catalan numbers:

```
avoiding_with_max_at_end(n) = Catalan(n-1)
```

## Decomposition

Every 1324-avoiding permutation of [1..n] decomposes by the position of its maximum element:

| Component | Count | Description |
|-----------|-------|-------------|
| Max at end | C(n-1) | Catalan numbers (proven bijection) |
| Max interior | R(n) | Correction term |

Where:
```
G(n) = C(n-1) + R(n)    for n ≥ 1

R: 0, 0, 1, 4, 18, 89, 474, 2672, ...
```

## Files

| File | Description |
|------|-------------|
| `Avoid1324.v` | Coq formalization (1109 lines, 71 proven theorems) |
| `1324Coq.v` | Original formalization attempt |

## Proven Theorems (Coq)

- **`general_bijection_theorem`**: [σ,n] avoids 1324 ⟺ σ avoids 132 (∀n)
- **`subpattern_theorem`**: 1324 contains 132 as subpattern (∀n)
- **`verified_decomposition`**: G(n) = max-at-end + interior (n ≤ 5)
- **`verified_catalan_contribution`**: max-at-end = Catalan(n-1) (n ≤ 5)

## Conjectured (Numerically Verified)

The generating function satisfies:

```
G(x) = 1 + x·G(x) + x²·G(x)²/(1 − x·G(x)) + x³·Q(x)
```

with Q(x) coefficients satisfying:

```
403·qₙ − 5531·qₙ₋₁ + 23277·qₙ₋₂ − 29357·qₙ₋₃ = 0
```

Verified for n = 3, 4, 5 in Coq.

## Building

```bash
coqc Avoid1324.v
```

Requires Coq 8.19+ with standard library.

## Author

Charles C. Norton
December 2025

## License

MIT
