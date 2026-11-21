# Second-stage quality verification

`prepare_quality_review.py` validates original source pairing, both raw outputs, EOS, and the prior blind mapping, then creates private review packets and empty scores in a new output directory. Run once in a fresh experiment copy with the original private pilot files; it refuses to overwrite review data. It never generates scores. `quality_audit.json` contains only non-sensitive aggregate findings and a generalized unique-ASK analysis. Review context contains medical text and is not guaranteed anonymized; inspect and de-identify it before any authorized sharing. The private key must remain separate until rating is complete.

The original report and metrics remain the Phase 1 measurements. No clinical claim or new model generation is made in Phase 2. The existing parent `SHA256SUMS` is regenerated after the report update, and this directory has its own checksum manifest.
