# Report tables

| system | Acc. | Macro-F1 | Evid. F1 | ECE | Gold removed: NEI% | Control: NEI% | Partial: acc. | Partial: NEI% | Silent gold: NEI% | Silent control: NEI% | Silent partial: acc. | Unsupported | Unnecessary abstention | Polarity | ms/claim |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| d1-3B zero-shot | 78.7 | 77.0 | 40.5 | 7.0 | 38.6 | 13.5 | 69.5 | 17.0 | 38.2 | 14.0 | 68.8 | 23.0 | 25.0 | 23.0 | 150.5 |
| NLI DeBERTa-v3 | 76.6 | 75.4 | 47.4 | 17.1 | 48.8 | 18.8 | 66.7 | 24.1 | 48.8 | 18.8 | 66.7 | 24.0 | 34.0 | 20.0 | 29.6 |
| MultiVerS (fine-tuned) | 83.8 | 82.5 | 71.5 | 9.6 | 35.7 | 7.7 | 65.2 | 17.0 | 35.7 | 7.7 | 65.2 | 16.0 | 14.0 | 24.0 | 43.1 |
| Qwen3.8-27B | 83.8 | 83.0 | 68.3 | 9.4 | 47.8 | 8.7 | 78.7 | 14.2 | 48.8 | 8.2 | 78.0 | 21.0 | 18.0 | 15.0 | 1266.3 |
| Original SFT (final run) | 86.5 | 85.4 | 76.3 | 12.0 | 44.9 | 12.1 | 74.5 | 15.6 | 44.9 | 12.6 | 74.5 | 10.0 | 21.0 | 14.0 | 325.7 |
| Augmented SFT (final run) | 86.2 | 85.1 | 76.7 | 12.9 | 78.3 | 13.5 | 54.6 | 36.2 | 67.6 | 12.1 | 66.0 | 10.0 | 21.0 | 15.0 | 325.9 |
| Paired hinge (final run) | 85.3 | 84.3 | 77.0 | 12.8 | 83.1 | 17.4 | 60.3 | 33.3 | 75.4 | 15.0 | 68.1 | 6.0 | 30.0 | 13.0 | 325.9 |
| Original SFT (3 seeds) | 85.4 ± 0.3 | 84.3 ± 0.3 | 76.3 ± 1.2 | 13.0 ± 0.8 | 48.3 ± 3.4 | 12.1 ± 2.2 | 74.5 ± 3.3 | 16.5 ± 1.5 | 49.6 ± 4.1 | 13.0 ± 1.4 | 73.5 ± 3.6 | 10.0 ± 3.0 | 22.0 ± 3.6 | 16.7 ± 0.6 | 325.9 ± 0.1 |
| Augmented SFT (3 seeds) | 85.0 ± 0.5 | 84.0 ± 0.8 | 75.8 ± 0.7 | 12.6 ± 0.5 | 82.3 ± 1.5 | 17.6 ± 0.6 | 57.4 ± 2.6 | 36.9 ± 2.6 | 72.8 ± 3.3 | 14.5 ± 1.9 | 67.6 ± 2.2 | 12.0 ± 2.6 | 26.3 ± 3.1 | 11.7 ± 1.2 | 325.9 ± 0.6 |
| Paired hinge (3 seeds) | 85.8 ± 0.6 | 84.8 ± 0.8 | 76.2 ± 0.5 | 12.1 ± 0.6 | 82.4 ± 1.0 | 16.9 ± 1.4 | 60.0 ± 2.5 | 34.0 ± 1.4 | 72.5 ± 2.5 | 13.2 ± 1.8 | 68.3 ± 2.5 | 11.7 ± 2.3 | 21.3 ± 4.0 | 14.3 ± 1.5 | 326.3 ± 0.4 |

Paired bootstrap over claims (2000 resamples), seed-averaged difference [95% CI]:

- Augmented SFT vs Original SFT: macro-F1 -0.2 [-2.4, +2.0]; gold-removed NEI% +34.0 [+28.7, +39.8]; control NEI% +5.5 [+2.9, +8.4]; partial acc. -17.0 [-23.4, -10.6]
- Paired hinge vs Original SFT: macro-F1 +0.5 [-1.7, +2.8]; gold-removed NEI% +34.1 [+28.7, +39.9]; control NEI% +4.8 [+2.3, +7.6]; partial acc. -14.4 [-20.8, -8.5]
- Paired hinge vs Augmented SFT: macro-F1 +0.7 [-0.7, +2.2]; gold-removed NEI% +0.2 [-1.6, +1.9]; control NEI% -0.6 [-2.7, +1.3]; partial acc. +2.6 [-0.7, +5.9]
