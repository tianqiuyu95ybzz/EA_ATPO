# Expert A training status

**NOT STARTED.** The prerequisites are incomplete: 291 review rows are pending; the complete provenance-verified Base outputs are missing; no GPU currently passes the no-sharing gate; and the 10-step AWQ LoRA forward/backward/checkpoint/reload/inference smoke has not run. No A adapter, optimizer state, epoch checkpoint, loss, selection threshold or accuracy exists for V3.

The independent A configuration fixes the input boundary to public conversation and Base answer, the JSON actions KEEP/REVISE/ABSTAIN, action and correction weights 1.0, correction loss only on REVISE, and a maximum of 10 epochs. KEEP/ABSTAIN must programmatically copy the original Base response. These settings are preparation only: the actual Base-aware SFT training implementation and full model-level validation remain to be completed after the hard gates pass. No legacy training run is attributed to V3.
