from pathlib import Path

BASELINE_ID = "qwen3_8b_ultrafeedback_margin_consumers_v1"
VARIANTS = ("dpo", "mmpo", "odpo_loggap", "scaled_dpo_gap_transfer")

RAW_REVISION = "40b436560ca83a8dba36114c22ab3c66e43f6d5e"
H4_REVISION = "3949bf5f8c17c394422ccfab0c31ea9c20bdeb85"
MODEL_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
TRL_COMMIT = "accf7383a33c618a2edc7205107120b5f32e28a3"

DPO_COMMIT = "f8b8c0f49dc92a430bae41585f9d467d3618fe2f"
MMPO_COMMIT = "ef3a91dfe1707ce09d4442b0a13285f87f9ae636"
ODPO_COMMIT = "6152f67c8cddc223ca5affc7e261d967568068ee"

ASPECTS = ("instruction_following", "truthfulness", "honesty", "helpfulness")
QWEN_PAD_ID = 151643
QWEN_EOS_ID = 151645
MAX_PROMPT_LENGTH = 1024
MAX_COMPLETION_LENGTH = 1024
MAX_LENGTH = 2048

EXPECTED = {
    "train_source_rows": 61135,
    "test_source_rows": 2000,
    "train_positive": 38263,
    "train_exact_tie": 7173,
    "train_direction_conflict": 11309,
    "train_invalid_rating": 4321,
    "train_ambiguous_alignment": 69,
    "train_positive_dedup_drop": 5,
    "train": 36258,
    "validation": 2000,
    "test": 1228,
    "train_normalized_one": 1830,
    "validation_normalized_one": 104,
    "test_normalized_one": 60,
}

DEFAULT_ROOT = Path("/root/autodl-tmp/rubric")
