import torch
import safetensors.torch as st
from qwen_vl_utils import process_vision_info
from transformers import Qwen3VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor

messages_generate = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "asset/road_sign_asphalt_road.jpg"},
            {"type": "text", "text": "Briefly describe this image."},
        ]
    }
]

def gen_test_case(messages, save_name, processor, model, save_dir="asset/output"):
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=16 if "Qwen3-VL" in model_name else 14,
        return_video_metadata=True,
        return_video_kwargs=True,
    )

    if video_inputs is not None:
        video_inputs, video_metadatas = zip(*video_inputs)
        video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)
    else:
        video_metadatas = None
        
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        video_metadata=video_metadatas,
        padding=True,
        do_resize=False,
        return_tensors="pt",
        **video_kwargs,
    )

    inputs = inputs.to(model.device)

    # Generate model output with hidden states for visualization
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=1.0,
        top_p=1.0,
        top_k=3,
        use_cache=True,
        output_scores=True,
        output_hidden_states=True, # ---> TAM needs hidden states
        return_dict_in_generate=True
    )

    generated_ids = outputs.sequences
    hidden_state = torch.cat([feats[-1] for feats in outputs.hidden_states], dim=1)

    generated = processor.decode(outputs.sequences[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    messages.append({"role": "assistant", "content": [{"type": "text", "text": generated}]})

    candidate_ids_per_step = torch.cat([
                step_scores[0].topk(3).indices.unsqueeze(0)
                for step_scores in outputs.scores
    ], dim=0)

    os.makedirs(save_dir, exist_ok=True)

    import json
    with open(f"{save_dir}/{save_name}.json", "w") as f:
        json.dump(messages, f)

    st.save_file({
        "hidden_state": hidden_state,
        "generated_ids": generated_ids,
        "candidate_ids_per_step": candidate_ids_per_step,
        "image_grid_thw": inputs.get("image_grid_thw", torch.empty(0)),
        "video_grid_thw": inputs.get("video_grid_thw", torch.empty(0)),
        }, f"{save_dir}/{save_name}.safetensors")

    return text, inputs, outputs

if __name__ == "__main__":
    import os
    model_name = os.environ.get("MODEL_PATH", "Qwen/Qwen3-VL-2B-Instruct")

    assert "Qwen3-VL" in model_name

    model_class = Qwen3VLForConditionalGeneration if "Qwen3-VL" in model_name else Qwen2_5_VLForConditionalGeneration
    model = model_class.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_name)

    gen_test_case(messages_generate, "generate", processor, model)