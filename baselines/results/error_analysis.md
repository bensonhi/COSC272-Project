## Label errors (count; of which confidence >= 0.8)

| system | unsupported judgement | unnecessary abstention | polarity confusion | correct S/C with missed gold sentence | ... with extra selected sentence |
|---|---|---|---|---|---|
| b0_label_prior | 126 (0) | 0 (0) | 70 (0) | 137/137 | 0/137 |
| b1_d1_zero_shot | 23 (5) | 25 (2) | 23 (5) | 42/159 | 119/159 |
| b2_d1_original_sft_seed42 | 15 (11) | 13 (13) | 19 (19) | 69/175 | 49/175 |
| b2_d1_original_sft_seed44 | 9 (7) | 17 (16) | 16 (15) | 68/174 | 55/174 |
| b3_nli_zero_shot_abstract | 24 (16) | 34 (26) | 20 (12) | 101/153 | 61/153 |
| b3_nli_zero_shot_sentence | 39 (34) | 57 (45) | 31 (26) | 62/119 | 53/119 |
| b4_multivers_fever_sci | 10 (7) | 71 (58) | 27 (18) | 97/109 | 2/109 |
| b4b_multivers_finetuned | 16 (7) | 14 (10) | 24 (15) | 80/169 | 58/169 |
| b5_llm_qwen3.8_27b | 21 (13) | 18 (9) | 15 (8) | 83/174 | 42/174 |

## Accuracy by claim subgroup (n; accuracy; high-confidence errors)

| system | numeric | causal | negated | several abstracts |
|---|---|---|---|---|
| b0_label_prior | 53; 0.38; 0 | 138; 0.42; 0 | 40; 0.33; 0 | 22; 0.45; 0 |
| b1_d1_zero_shot | 53; 0.79; 3 | 138; 0.84; 7 | 40; 0.75; 4 | 22; 0.82; 0 |
| b2_d1_original_sft_seed42 | 53; 0.83; 8 | 138; 0.89; 15 | 40; 0.88; 4 | 22; 0.91; 2 |
| b2_d1_original_sft_seed44 | 53; 0.81; 10 | 138; 0.90; 13 | 40; 0.88; 4 | 22; 0.86; 3 |
| b3_nli_zero_shot_abstract | 53; 0.72; 10 | 138; 0.73; 24 | 40; 0.80; 5 | 22; 0.91; 2 |
| b3_nli_zero_shot_sentence | 53; 0.66; 15 | 138; 0.57; 50 | 40; 0.50; 16 | 22; 0.59; 7 |
| b4_multivers_fever_sci | 53; 0.83; 7 | 138; 0.64; 40 | 40; 0.65; 12 | 22; 0.82; 3 |
| b4b_multivers_finetuned | 53; 0.81; 5 | 138; 0.83; 15 | 40; 0.85; 2 | 22; 0.91; 0 |
| b5_llm_qwen3.8_27b | 53; 0.81; 4 | 138; 0.81; 17 | 40; 0.75; 6 | 22; 0.86; 1 |
