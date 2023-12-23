import torch
import gradio
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer


model_name = "mistralai/Mistral-7B-Instruct-v0.2"
torch_dtype = torch.bfloat16  # Set the appropriate torch data type
device = "cuda"

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=model_name,
    trust_remote_code=True,
    device_map=device,
    torch_dtype=torch_dtype,
)

tokenizer = AutoTokenizer.from_pretrained(model_name, device_map=device, trust_remote_code=True)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
tokenizer.padding_side = 'right'  # TRL requires right padding

def run_generation(user_text, top_p, temperature, top_k, max_new_tokens):
    template = "<s>[INST] {} [/INST]"
    model_inputs = tokenizer(template.format(user_text), return_tensors="pt")
    model_inputs = model_inputs.to(device)

    # Generate text in a separate thread
    streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=False, skip_special_tokens=False)

    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p,
        temperature=float(temperature),
        top_k=top_k,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    # Retrieve and yield the generated text
    model_output = ""
    for new_text in streamer:
        model_output += new_text
        yield model_output
    return model_output


with gradio.Blocks() as demo:
    with gradio.Row():
        with gradio.Column(scale=4):
            user_text = gradio.Textbox(placeholder="Write your question here", label="User input")
            model_output = gradio.Textbox(label="Model output", lines=10, interactive=False)
            button_submit = gradio.Button(value="Submit")

        with gradio.Column(scale=1):
            max_new_tokens = gradio.Slider(minimum=1, maximum=1000, value=250, step=1, label="Max New Tokens")
            top_p = gradio.Slider(minimum=0.05, maximum=1.0, value=0.95, step=0.05, label="Top-p (nucleus sampling)")
            top_k = gradio.Slider(minimum=1, maximum=50, value=50, step=1, label="Top-k")
            temperature = gradio .Slider(minimum=0.1, maximum=5.0, value=0.8, step=0.1, label="Temperature")

    user_text.submit(run_generation, [user_text, top_p, temperature, top_k, max_new_tokens], model_output)
    button_submit.click(run_generation, [user_text, top_p, temperature, top_k, max_new_tokens], model_output)

    demo.queue(max_size=32).launch()
