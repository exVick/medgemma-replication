"""
Central prompt registry.

Keep all prompts in one place for:
- reproducibility
- easy ablations
- cleaner experiment modules
"""

PROMPTS = {
    # Report generation
    ### separated both sections:
    "findings": (
        "You are provided with a chest X-ray image. "
        "Write the findings section of the radiology report for this chest X-ray."
    ),
    "impression": (
        "You are provided with a chest X-ray image. "
        "Write the impression section of the radiology report for this chest X-ray."
    ),
    ### original paper prompt:
    "findings_and_impression": (
        "{indication} findings:"
    ),

    # Classification
    ### baseline prompt:
    # "classify_condition": (
    #     "Does this chest X-ray show {condition}? Answer yes or no."
    # ),
    ### original paper prompt:
    "classify_condition": (
        "Is there {condition} in this image? You may write out your argument before stating your final very short, definitive, and concise answer (if possible, a single word or the letter corresponding to your answer choice) X in the format 'Final Answer: X':"
    ),

    # TODO (future): add VQA prompt templates
    # "vqa": "Answer the medical question about this image: {question}"
}