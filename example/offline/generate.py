import os

import torch
import safetensors.torch as st

from tamx.qwen import build_qwen_inputs, load_qwen_model_bundle

messages_generate = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "asset/road_sign_asphalt_road.jpg"},
            {"type": "text", "text": "Briefly describe this image."},
        ]
    }
]

def gen_test_case(messages, save_name, processor, model, model_info, save_dir="asset/output"):
    prompt_text, inputs, _, _, _, _ = build_qwen_inputs(
        processor=processor,
        model_info=model_info,
        messages=messages,
        add_generation_prompt=True,
        enable_thinking=False,
        device=model.device,
    )

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

    print(f"Generated: {generated}")
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
    print(f"Saved to {save_dir}/{save_name}.safetensors")

    return prompt_text, inputs, outputs

if __name__ == "__main__":
    model_name = os.environ.get("MODEL_PATH", "/data/zoo/Qwen3.5-2B")

    model, processor, model_info = load_qwen_model_bundle(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )

    gen_test_case(messages_generate, "generate", processor, model, model_info)
