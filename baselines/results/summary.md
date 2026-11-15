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
| d1_augmented_sft_seed43 | 0.837 [0.795, 0.877] | 0.847 | 0.865 / 0.862 / 0.783 | 0.765 (0.80 / 0.73) | 0.790 | 0.821 | 0.937 | 0.128 | 323 |
| d1_cf_all_seed42 | 0.830 [0.786, 0.873] | 0.841 | 0.856 / 0.873 / 0.760 | 0.761 (0.80 / 0.72) | 0.784 | 0.832 | 0.932 | 0.145 | 324 |
| d1_cf_all_seed43 | 0.840 [0.796, 0.880] | 0.853 | 0.872 / 0.883 / 0.764 | 0.762 (0.80 / 0.72) | 0.784 | 0.840 | 0.928 | 0.133 | 323 |
| d1_cf_evidence_only_seed42 | 0.847 [0.805, 0.887] | 0.856 | 0.868 / 0.876 / 0.797 | 0.758 (0.80 / 0.72) | 0.782 | 0.822 | 0.908 | 0.125 | 325 |
| d1_cf_evidence_only_seed43 | 0.835 [0.791, 0.873] | 0.844 | 0.859 / 0.866 / 0.779 | 0.762 (0.81 / 0.72) | 0.790 | 0.828 | 0.908 | 0.138 | 324 |
| d1_cf_evidence_only_seed44 | 0.841 [0.800, 0.879] | 0.850 | 0.876 / 0.857 / 0.789 | 0.752 (0.80 / 0.71) | 0.775 | 0.821 | 0.918 | 0.139 | 322 |
| final_d1_augment | 0.851 [0.812, 0.891] | 0.862 | 0.882 / 0.876 / 0.794 | 0.768 (0.81 / 0.73) | 0.792 | 0.828 | 0.908 | 0.130 | 562 |
| final_d1_augment_pair | 0.846 [0.800, 0.885] | 0.856 | 0.873 / 0.867 / 0.797 | 0.768 (0.83 / 0.72) | 0.789 | 0.844 | 0.918 | 0.125 | 562 |
| final_d1_original | 0.851 [0.808, 0.889] | 0.862 | 0.882 / 0.879 / 0.791 | 0.763 (0.82 / 0.71) | 0.786 | 0.835 | 0.942 | 0.119 | 562 |

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
| d1_augmented_sft_seed43 | 0.821 | 0.802 | 0.805 | 0.768 | 0.169 | +0.633 |
| d1_cf_all_seed42 | 0.807 | 0.797 | 0.800 | 0.778 | 0.159 | +0.638 |
| d1_cf_all_seed43 | 0.812 | 0.802 | 0.799 | 0.778 | 0.159 | +0.643 |
| d1_cf_evidence_only_seed42 | 0.821 | 0.792 | 0.788 | 0.773 | 0.159 | +0.633 |
| d1_cf_evidence_only_seed43 | 0.812 | 0.807 | 0.796 | 0.783 | 0.150 | +0.657 |
| d1_cf_evidence_only_seed44 | 0.802 | 0.807 | 0.806 | 0.754 | 0.174 | +0.633 |
| final_d1_augment | 0.826 | 0.783 | 0.787 | 0.792 | 0.135 | +0.647 |
| final_d1_augment_pair | 0.797 | 0.831 | 0.824 | 0.758 | 0.174 | +0.657 |
| final_d1_original | 0.826 | 0.449 | 0.444 | 0.812 | 0.121 | +0.329 |

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
| d1_augmented_sft_seed43 | 0.821 | 0.652 | 0.633 | 0.812 | 0.121 | +0.531 |
| d1_cf_all_seed42 | 0.807 | 0.725 | 0.715 | 0.792 | 0.130 | +0.594 |
| d1_cf_all_seed43 | 0.812 | 0.749 | 0.752 | 0.792 | 0.150 | +0.599 |
| d1_cf_evidence_only_seed42 | 0.821 | 0.729 | 0.720 | 0.797 | 0.140 | +0.589 |
| d1_cf_evidence_only_seed43 | 0.812 | 0.676 | 0.680 | 0.787 | 0.135 | +0.541 |
| d1_cf_evidence_only_seed44 | 0.802 | 0.734 | 0.735 | 0.773 | 0.155 | +0.580 |
| final_d1_augment | 0.826 | 0.676 | 0.670 | 0.802 | 0.121 | +0.556 |
| final_d1_augment_pair | 0.797 | 0.754 | 0.739 | 0.768 | 0.150 | +0.604 |
| final_d1_original | 0.826 | 0.449 | 0.450 | 0.812 | 0.126 | +0.324 |

Counterfactual minimal pairs of the test claims (LLM-edited, verified by the Qwen3.8-27B judge, so B5 is excluded: it agrees with every kept edit by construction). acc = accuracy on the edited input; pair = correct on both the original and the edited input.

| system | claim_negation acc / pair (n) | evidence_flip acc / pair (n) | evidence_weaken acc / pair (n) |
|---|---|---|---|
| b1_d1_zero_shot | 0.826 / 0.717 (138) | 0.758 / 0.641 (128) | 0.357 / 0.143 (14) |
| b2_d1_original_sft_seed42 | 0.913 / 0.870 (138) | 0.844 / 0.812 (128) | 0.214 / 0.143 (14) |
| b2_d1_original_sft_seed43 | 0.891 / 0.848 (138) | 0.828 / 0.805 (128) | 0.286 / 0.143 (14) |
| b2_d1_original_sft_seed44 | 0.920 / 0.891 (138) | 0.781 / 0.734 (128) | 0.357 / 0.286 (14) |
| b3_nli_zero_shot_abstract | 0.812 / 0.746 (138) | 0.672 / 0.602 (128) | 0.571 / 0.214 (14) |
| b4_multivers_fever_sci | 0.384 / 0.239 (138) | 0.508 / 0.336 (128) | 0.571 / 0.286 (14) |
| b4b_multivers_finetuned | 0.855 / 0.797 (138) | 0.750 / 0.672 (128) | 0.143 / 0.143 (14) |
| d1_augmented_sft_seed42 | 0.891 / 0.841 (138) | 0.750 / 0.727 (128) | 0.500 / 0.357 (14) |
| d1_augmented_sft_seed43 | 0.884 / 0.841 (138) | 0.664 / 0.617 (128) | 0.357 / 0.214 (14) |
| d1_cf_all_seed42 | 0.899 / 0.870 (138) | 0.914 / 0.867 (128) | 0.786 / 0.643 (14) |
| d1_cf_all_seed43 | 0.913 / 0.862 (138) | 0.883 / 0.859 (128) | 0.857 / 0.643 (14) |
| d1_cf_evidence_only_seed42 | 0.891 / 0.855 (138) | 0.859 / 0.828 (128) | 0.857 / 0.786 (14) |
| d1_cf_evidence_only_seed43 | 0.906 / 0.870 (138) | 0.906 / 0.867 (128) | 0.786 / 0.714 (14) |
| d1_cf_evidence_only_seed44 | 0.899 / 0.841 (138) | 0.883 / 0.836 (128) | 0.857 / 0.714 (14) |
| final_d1_augment | 0.913 / 0.870 (138) | 0.617 / 0.594 (128) | 0.357 / 0.286 (14) |
| final_d1_augment_pair | 0.855 / 0.804 (138) | 0.547 / 0.531 (128) | 0.500 / 0.357 (14) |
| final_d1_original | 0.884 / 0.841 (138) | 0.773 / 0.750 (128) | 0.286 / 0.143 (14) |

Out-of-distribution: HealthVer test (1,823 human-labelled COVID-19 claim / evidence-passage pairs, no training on HealthVer).

| system | macro-F1 [95% CI] | acc | F1 NEI / SUP / CON | ECE |
|---|---|---|---|---|
| b1_d1_zero_shot | 0.488 [0.463, 0.512] | 0.546 | 0.638 / 0.407 / 0.419 | 0.235 |
| b2_d1_original_sft_seed42 | 0.422 [0.397, 0.445] | 0.517 | 0.630 / 0.392 / 0.243 | 0.460 |
| b2_d1_original_sft_seed43 | 0.417 [0.392, 0.440] | 0.516 | 0.626 / 0.395 / 0.229 | 0.454 |
| b2_d1_original_sft_seed44 | 0.435 [0.409, 0.461] | 0.523 | 0.630 / 0.404 / 0.270 | 0.457 |
| d1_augmented_sft_seed42 | 0.349 [0.328, 0.372] | 0.477 | 0.606 / 0.294 / 0.148 | 0.507 |
| d1_augmented_sft_seed43 | 0.366 [0.343, 0.390] | 0.485 | 0.611 / 0.316 / 0.172 | 0.492 |
| d1_cf_all_seed42 | 0.343 [0.321, 0.366] | 0.471 | 0.604 / 0.257 / 0.168 | 0.516 |
| d1_cf_all_seed43 | 0.386 [0.361, 0.410] | 0.493 | 0.617 / 0.300 / 0.241 | 0.493 |
| d1_cf_evidence_only_seed42 | 0.386 [0.360, 0.410] | 0.495 | 0.617 / 0.322 / 0.218 | 0.482 |
| d1_cf_evidence_only_seed43 | 0.398 [0.372, 0.422] | 0.501 | 0.620 / 0.320 / 0.254 | 0.482 |
| d1_cf_evidence_only_seed44 | 0.365 [0.341, 0.388] | 0.485 | 0.612 / 0.300 / 0.181 | 0.495 |
| final_d1_augment | 0.361 [0.337, 0.384] | 0.485 | 0.611 / 0.311 / 0.160 | 0.503 |
| final_d1_augment_pair | 0.340 [0.318, 0.362] | 0.474 | 0.605 / 0.280 / 0.137 | 0.514 |
| final_d1_original | 0.410 [0.385, 0.436] | 0.511 | 0.625 / 0.365 / 0.241 | 0.465 |
