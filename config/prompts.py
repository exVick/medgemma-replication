"""
Central prompt registry.

Keep all prompts in one place for:
- reproducibility
- easy ablations
- cleaner experiment modules
"""

PROMPTS = {
    # Report generation
    "findings": (
        "You are provided with a chest X-ray image. "
        "Write the findings section of the radiology report for this chest X-ray."
    ),
    "impression": (
        "You are provided with a chest X-ray image. "
        "Write the impression section of the radiology report for this chest X-ray."
    ),

    # Classification
    "classify_condition": (
        "Does this chest X-ray show {condition}? Answer yes or no."
    ),

    # TODO (future): add VQA prompt templates
    # "vqa": "Answer the medical question about this image: {question}"
}