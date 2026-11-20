# Oracle upper bounds

There is no V3 scored dataset yet, so no empirical Oracle result is reported.

- **Oracle Expert:** on cases with approved, reliable Gold, hypothetically KEEP when Base is correct and use Gold to correct every Base error. This is an idealized theoretical expert.
- **Oracle Selector:** for each fixed Base and Expert output, hypothetically select a correct one using Gold; an instance where both are wrong stays wrong. This measures a fixed-output selection upper bound and is generally below the idealized Oracle Expert.

Neither hypothetical decision enters SFT, Router training, test-time features or Dev/Test threshold optimization. `scripts/evaluate_oracle.py` extracts these bounds only from a completed offline aggregate; its availability is not evidence that evaluation has occurred.
