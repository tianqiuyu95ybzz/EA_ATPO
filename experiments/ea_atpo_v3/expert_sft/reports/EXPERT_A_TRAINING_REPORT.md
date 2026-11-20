# Expert A training status

**NOT STARTED.** All 291 review rows remain pending. The complete provenance-verified Base outputs have been generated and the separate 10-step AWQ LoRA smoke passed, with its loading warning recorded in `AWQ_LORA_SMOKE_STATUS.md`. No A adapter, optimizer state, epoch checkpoint, loss, selection threshold or accuracy exists for V3.

The independent A configuration fixes the input boundary to public conversation and Base answer, the JSON actions KEEP/REVISE/ABSTAIN, action and correction weights 1.0, correction loss only on REVISE, and a maximum of 10 epochs. KEEP/ABSTAIN must programmatically copy the original Base response. These settings are preparation only: the actual Base-aware SFT training implementation and full model-level validation remain to be completed after the hard gates pass. No legacy training run is attributed to V3.
