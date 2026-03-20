from typing import Optional, Tuple, Dict, Any

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig


def load_model(
    model_id: str = "google/medgemma-4b-it",
    float_type: Optional[str] = "bfloat16",
    use_8bit: bool = False,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Shared model loader for all experiments.
    Returns (model, processor, experiment_meta_partial).
    """
    quant_config = None
    if use_8bit:
        quant_config = BitsAndBytesConfig(load_in_8bit=True)
        float_type = None

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        dtype=float_type,
        quantization_config=quant_config,
        device_map={"": 0},  # explicit single visible GPU - 'cuda' might be ambiguous 
    )
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()

    vram_after_load = (
        torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
    )

    meta = {
        "model_id": model_id,
        "float_type": str(float_type),
        "use_8bit": use_8bit,
        "vram_after_load_gb": round(vram_after_load, 3) if vram_after_load else None,
    }

    print(
        f"Loaded '{model_id}' | dtype={float_type} | 8-bit={use_8bit}"
        + (f" | VRAM after load: {vram_after_load:.2f} GB" if vram_after_load else "")
    )

    return model, processor, meta


def run_inference(
    model,
    processor,
    image: Image.Image,
    prompt_text: str,
    max_new_tokens: int = 512,
    do_sample: bool = False,
) -> str:
    """
    Generic single-image inference used by all tasks.
    """
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are a helpful medical assistant."}
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt",
    ).to(model.device)

    # Match pixel dtype to model dtype
    model_dtype = next(model.parameters()).dtype
    inputs["pixel_values"] = inputs["pixel_values"].to(model_dtype)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )

    generated_ids = output_ids[0][input_len:]
    response = processor.decode(generated_ids, skip_special_tokens=True)
    return response.strip()