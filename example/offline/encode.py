import os

import torch
import safetensors.torch as st

from tamx.qwen import build_qwen_inputs, load_qwen_model_bundle

messages_encode = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "asset/road_sign_asphalt_road.jpg"},
            {"type": "text", "text": "Briefly describe this image."},
        ]
    },
    {
        "role": "assistant", "content": [
            {"type": "text", "text": "This is an image of a rural road sign in a countryside setting. A bright orange, square sign with black lettering that reads \"SLOW DOWN\" is prominently displayed on the right. The road is paved and curves gently into the distance, bordered by grassy hills and dense green trees. The sky is overcast with light clouds."}
        ]
    }
]

def gen_test_case(messages, save_name, processor, model, model_info, save_dir="asset/output"):
    _, inputs, _, _, _, _ = build_qwen_inputs(
        processor=processor,
        model_info=model_info,
        messages=messages,
        add_generation_prompt=False,
        enable_thinking=False,
        device=model.device,
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=1,
        temperature=1.0,
        top_p=1.0,
        output_hidden_states=True, # ---> TAM needs hidden states
        return_dict_in_generate=True
    )

    hidden_state = torch.cat([feats[-1] for feats in outputs.hidden_states], dim=1)[:, :-1, :]
    generated_ids = inputs["input_ids"]
    print(f"Generated IDs first 10: {generated_ids[0,:10]}")

    os.makedirs(save_dir, exist_ok=True)

    import json
    with open(f"{save_dir}/{save_name}.json", "w") as f:
        json.dump(messages, f)

    st.save_file({
        "hidden_state": hidden_state,
        "generated_ids": generated_ids,
        "image_grid_thw": inputs.get("image_grid_thw", torch.empty(0)),
        "video_grid_thw": inputs.get("video_grid_thw", torch.empty(0)),
        }, f"{save_dir}/{save_name}.safetensors"
    )
    print(f"Saved to {save_dir}/{save_name}.safetensors")


if __name__ == "__main__":
    model_name = os.environ.get("MODEL_PATH", "/data/zoo/Qwen3.5-2B")

    model, processor, model_info = load_qwen_model_bundle(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )

    gen_test_case(messages_encode, "encode", processor, model, model_info)
