Avoiding calibration measured under estimator v2.0, superseded by v2.1.

v2.1 bounds the normalised observation and drops non-finite gradient batches
(docs/v2/PREREG_v2.md, section 8, 2026-11-02 "bounded observations"). On
Avoiding the bound is not expected to engage, since the largest normalised
observation measured there was 6.0 against a bound of 50, but these rows were
produced before the bound existed and carry no diagnostic that would show it.
They are kept as a record and are not read by any analysis.

108 cells, all completed, zero exceptions. The weight the pre-registered rule
chose from them was 1.0 for each of mop, mi and bmd.
