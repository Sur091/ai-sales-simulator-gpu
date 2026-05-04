import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def quick_test():
    """Quick test to verify the model loads and generates responses."""

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2b",
        torch_dtype=torch.float16,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(model, "output/gemma2b-lora-windows")
    model.eval()

    # Test prompt - Life Insurance Sales Scenario
    test_prompt = """USER: You are Laura Thompson, a Retired Accountant at Schneider Electric in Manufacturing/Electrical.

Pain points: Concerned about financial security in retirement; Wants to protect family but worried about costs; Unsure about life insurance necessity.
Needs: Affordable life insurance options; Clear explanation of coverage; Peace of mind for beneficiaries.

Respond realistically as this customer in a sales conversation.

USER: Hi Laura, I'm calling from SecureLife Insurance. We help retirees like yourself protect their family's financial future. Have you thought about life insurance lately?

ASSISTANT:"""

    print("Generating response...")
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    # Stop at next turn indicators
    response = response.split("USER:")[0].split("ASSISTANT:")[0].strip()

    print(f"\nCustomer Response: {response}")

    print("\nModel test completed successfully!")

if __name__ == "__main__":
    quick_test()