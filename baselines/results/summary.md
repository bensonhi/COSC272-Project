| system | macro-F1 [95% CI] | acc | F1 NEI / SUP / CON | evid. F1 (P / R) | evid. F1 human | evid. AUPRC | top-1 hit | ECE | median ms/claim |
|---|---|---|---|---|---|---|---|---|---|
| b0_label_prior | 0.194 [0.177, 0.211] | 0.411 | 0.000 / 0.583 / 0.000 | 0.000 (0.00 / 0.00) | 0.000 | 0.160 | 0.029 | 0.000 | 0 |
| b1_d1_zero_shot | 0.770 [0.723, 0.817] | 0.787 | 0.811 / 0.822 / 0.676 | 0.405 (0.28 / 0.71) | 0.365 | 0.404 | 0.720 | 0.070 | 151 |
| b2_d1_original_sft_seed42 | 0.851 [0.814, 0.888] | 0.859 | 0.888 / 0.867 / 0.797 | 0.771 (0.80 / 0.75) | 0.788 | 0.841 | 0.957 | 0.131 | 324 |
| b2_d1_original_sft_seed43 | 0.853 [0.813, 0.892] | 0.862 | 0.879 / 0.881 / 0.797 | 0.786 (0.79 / 0.78) | 0.796 | 0.853 | 0.942 | 0.113 | 324 |
| b2_d1_original_sft_seed44 | 0.863 [0.825, 0.901] | 0.874 | 0.900 / 0.886 / 0.803 | 0.754 (0.79 / 0.72) | 0.770 | 0.843 | 0.937 | 0.113 | 324 |
| b3_nli_zero_shot_abstract | 0.754 [0.704, 0.802] | 0.766 | 0.779 / 0.791 / 0.691 | 0.474 (0.48 / 0.47) | 0.472 | 0.449 | 0.705 | 0.171 | 30 |
| b3_nli_zero_shot_sentence | 0.616 [0.563, 0.666] | 0.619 | 0.644 / 0.567 / 0.636 | 0.474 (0.48 / 0.47) | 0.472 | 0.449 | 0.705 | 0.303 | 16 |
| b4_multivers_fever_sci | 0.600 [0.549, 0.655] | 0.676 | 0.741 / 0.715 / 0.344 | 0.165 (0.91 / 0.09) | 0.194 | 0.583 | 0.792 | 0.272 | 44 |
| b4b_multivers_finetuned | 0.825 [0.784, 0.864] | 0.838 | 0.880 / 0.848 / 0.748 | 0.715 (0.74 / 0.69) | 0.746 | 0.805 | 0.894 | 0.096 | 43 |
| b5_llm_qwen3.8_27b | 0.830 [0.786, 0.868] | 0.838 | 0.843 / 0.872 / 0.775 | 0.683 (0.78 / 0.61) | 0.717 | 0.607 | 0.802 | 0.094 | 1266 |
| d1_augmented_sft_seed42 | 0.850 [0.810, 0.891] | 0.859 | 0.880 / 0.868 / 0.803 | 0.781 (0.83 / 0.73) | 0.805 | 0.827 | 0.932 | 0.126 | 323 |

Stress test on the 207 test claims with gold evidence, deleted sentences blanked (empty slots, the training format) (evidence_removed: all gold evidence sentences, target NEI; random_removed: as many non-evidence sentences, label unchanged).

| system | original acc | evidence_removed -> NEI | mean P(NEI) | random_removed acc | random_removed -> NEI | selectivity |
|---|---|---|---|---|---|---|
| b1_d1_zero_shot | 0.768 | 0.386 | 0.368 | 0.758 | 0.135 | +0.251 |
| b2_d1_original_sft_seed42 | 0.845 | 0.357 | 0.361 | 0.845 | 0.068 | +0.290 |
| b2_d1_original_sft_seed43 | 0.860 | 0.406 | 0.404 | 0.850 | 0.082 | +0.324 |
| b2_d1_original_sft_seed44 | 0.841 | 0.464 | 0.456 | 0.836 | 0.092 | +0.372 |
| b3_nli_zero_shot_abstract | 0.739 | 0.488 | 0.500 | 0.749 | 0.188 | +0.300 |
| b4_multivers_fever_sci | 0.527 | 0.560 | 0.576 | 0.536 | 0.357 | +0.203 |
| b4b_multivers_finetuned | 0.816 | 0.357 | 0.377 | 0.807 | 0.077 | +0.280 |
| b5_llm_qwen3.8_27b | 0.841 | 0.478 | 0.464 | 0.841 | 0.087 | +0.391 |
| d1_augmented_sft_seed42 | 0.816 | 0.802 | 0.807 | 0.783 | 0.159 | +0.643 |

Stress test (silent deletion) on the 207 test claims with gold evidence, deleted sentences cut out and the rest renumbered (no visible gap) (evidence_removed: all gold evidence sentences, target NEI; random_removed: as many non-evidence sentences, label unchanged).

| system | original acc | evidence_removed -> NEI | mean P(NEI) | random_removed acc | random_removed -> NEI | selectivity |
|---|---|---|---|---|---|---|
| b1_d1_zero_shot | 0.768 | 0.382 | 0.371 | 0.758 | 0.140 | +0.242 |
| b2_d1_original_sft_seed42 | 0.845 | 0.377 | 0.373 | 0.850 | 0.063 | +0.314 |
| b2_d1_original_sft_seed43 | 0.860 | 0.406 | 0.407 | 0.855 | 0.082 | +0.324 |
| b2_d1_original_sft_seed44 | 0.841 | 0.459 | 0.452 | 0.836 | 0.092 | +0.367 |
| b3_nli_zero_shot_abstract | 0.739 | 0.488 | 0.500 | 0.749 | 0.188 | +0.300 |
| b4_multivers_fever_sci | 0.527 | 0.560 | 0.576 | 0.536 | 0.357 | +0.203 |
| b4b_multivers_finetuned | 0.816 | 0.357 | 0.377 | 0.807 | 0.077 | +0.280 |
| b5_llm_qwen3.8_27b | 0.841 | 0.488 | 0.469 | 0.850 | 0.082 | +0.406 |
| d1_augmented_sft_seed42 | 0.816 | 0.729 | 0.728 | 0.802 | 0.130 | +0.599 |
