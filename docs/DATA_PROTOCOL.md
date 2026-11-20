# Data protocol

Split by `case_id`, keep all states of a case in one split, and never expose Gold, hidden patient facts or future turns to model inputs. The available 291-state Pilot is simulated, not a human patient transcript corpus. Human Gold review is pending; original historical labels are not automatically approved. The 400-case frozen holdout is not opened for development or checkpoint selection.

Protected dialogue, Gold-bearing data, raw predictions and checkpoints stay in access-controlled PFS. Git contains only reproducible code, aggregate reports and privacy-safe manifests. Historical tracked trajectories require separate protected-data review; no history rewrite is implied by this document.
