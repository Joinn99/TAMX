import torch
import safetensors.torch as st
from qwen_vl_utils import process_vision_info
from transformers import Qwen3VLForConditionalGeneration, Qwen2_5_VLForConditionalGeneration, AutoProcessor

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

def gen_test_case(messages, save_name, processor, model, save_dir="asset/output"):
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
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


if __name__ == "__main__":
    import os
    model_name = os.environ.get("MODEL_PATH", "Qwen/Qwen3-VL-2B-Instruct")

    assert "Qwen3-VL" in model_name

    model_class = Qwen3VLForConditionalGeneration if "Qwen3-VL" in model_name else Qwen2_5_VLForConditionalGeneration
    model = model_class.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_name)


    gen_test_case(messages_encode, "encode", processor, model)
