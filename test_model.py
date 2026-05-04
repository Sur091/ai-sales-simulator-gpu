import os
import torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
MODEL_NAME_OR_PATH = "google/gemma-2b"
ADAPTER_PATH = "output/gemma2b-lora-windows"
MAX_NEW_TOKENS = 512
TEMPERATURE = 0.7
TOP_P = 0.9

def load_model():
    """Load the fine-tuned model with LoRA adapters."""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_OR_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    print("Loading LoRA adapters...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)

    model.eval()
    return model, tokenizer

def format_conversation(customer_persona, conversation_history):
    """Format conversation for the model - matches training format."""
    system_prompt = f"""You are {customer_persona.get('name', 'a customer')}, a {customer_persona.get('role', 'decision-maker')} at {customer_persona.get('company', 'their company')} in {customer_persona.get('industry', 'their industry')}.

Pain points: {'; '.join(customer_persona.get('pain_points', [])[:3])}.
Needs: {'; '.join(customer_persona.get('needs', [])[:3])}.

Respond realistically as this customer in a sales conversation."""

    # Start with system prompt as first user message (matches training)
    messages = [f"USER: {system_prompt}"]

    # Add conversation history
    for msg in conversation_history:
        if msg['role'] == 'user':
            messages.append(f"USER: {msg['content']}")
        elif msg['role'] == 'assistant':
            messages.append(f"ASSISTANT: {msg['content']}")

    # End with ASSISTANT: to prompt model for next response
    return "\n\n".join(messages) + "\n\nASSISTANT: "

def generate_response(model, tokenizer, prompt, max_new_tokens=MAX_NEW_TOKENS):
    """Generate response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

    # Clean up response - stop at next USER: or ASSISTANT: if present
    response = response.split("USER:")[0].split("ASSISTANT:")[0].strip()

    return response

def test_model():
    """Interactive testing of the fine-tuned model."""
    print("Loading fine-tuned Gemma model...")
    model, tokenizer = load_model()
    print("Model loaded successfully!\n")

    # Sample customer personas for testing - Life Insurance Sales
    test_personas = [
        {
            "name": "Laura Thompson",
            "role": "Retired Accountant",
            "company": "Schneider Electric",
            "industry": "Manufacturing/Electrical",
            "pain_points": ["Concerned about financial security in retirement", "Wants to protect family but worried about costs", "Unsure about life insurance necessity"],
            "needs": ["Affordable life insurance options", "Clear explanation of coverage", "Peace of mind for beneficiaries"]
        },
        {
            "name": "Jim Rodriguez",
            "role": "Retired Veteran",
            "company": "City Bus Service",
            "industry": "Transportation",
            "pain_points": ["Friends recommend life insurance but unsure about value", "Concerned about monthly premiums being too high", "Wants to provide for family but budget-conscious"],
            "needs": ["Budget-friendly life insurance plans", "Understanding of coverage options", "Flexible payment terms"]
        }
    ]

    print("Available test scenarios:")
    for i, persona in enumerate(test_personas, 1):
        print(f"{i}. {persona['name']} - {persona['role']} at {persona['company']}")

    while True:
        try:
            choice = input("\nSelect a scenario (1-2) or 'q' to quit: ").strip()

            if choice.lower() == 'q':
                break

            if not choice.isdigit() or int(choice) not in [1, 2]:
                print("Invalid choice. Please select 1 or 2.")
                continue

            persona = test_personas[int(choice) - 1]
            conversation_history = []

            print(f"\n--- Testing with {persona['name']} ---")
            print(f"Role: {persona['role']} at {persona['company']}")
            print(f"Industry: {persona['industry']}")
            print(f"Pain Points: {', '.join(persona['pain_points'])}")
            print(f"Needs: {', '.join(persona['needs'])}")
            print("\n" + "="*50)

            while True:
                user_input = input("\nSales Rep: ").strip()
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break

                # Add user message to history
                conversation_history.append({"role": "user", "content": user_input})

                # Format prompt and generate response
                prompt = format_conversation(persona, conversation_history)
                response = generate_response(model, tokenizer, prompt)

                print(f"\nCustomer ({persona['name']}): {response}")

                # Add assistant response to history
                conversation_history.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

def run_batch_test():
    """Run batch testing with predefined scenarios."""
    print("Loading model for batch testing...")
    model, tokenizer = load_model()

    test_scenarios = [
        {
            "persona": {
                "name": "Laura Thompson",
                "role": "Retired Accountant",
                "company": "Schneider Electric",
                "industry": "Manufacturing/Electrical",
                "pain_points": ["Concerned about financial security in retirement", "Wants to protect family but worried about costs"],
                "needs": ["Affordable life insurance options", "Clear explanation of coverage"]
            },
            "conversation": [
                {"role": "user", "content": "Hi Laura, I'm calling from SecureLife Insurance. We specialize in life insurance for retirees. How are you doing today?"},
                {"role": "user", "content": "That's great to hear. Many of our retired clients find peace of mind knowing their family is financially protected. Have you considered life insurance for your retirement planning?"}
            ]
        }
    ]

    print("\nRunning batch tests...\n")

    for i, scenario in enumerate(test_scenarios, 1):
        print(f"Test Scenario {i}:")
        print(f"Customer: {scenario['persona']['name']} ({scenario['persona']['role']})")
        print("-" * 50)

        for j, msg in enumerate(scenario['conversation'], 1):
            if msg['role'] == 'user':
                prompt = format_conversation(scenario['persona'], scenario['conversation'][:j])
                response = generate_response(model, tokenizer, prompt)

                print(f"Sales Rep: {msg['content']}")
                print(f"Customer: {response}")
                print()

if __name__ == "__main__":
    print("Gemma Fine-tuned Model Tester")
    print("=" * 40)

    while True:
        print("\nOptions:")
        print("1. Interactive testing")
        print("2. Batch testing")
        print("3. Quit")

        choice = input("\nSelect option (1-3): ").strip()

        if choice == "1":
            test_model()
        elif choice == "2":
            run_batch_test()
        elif choice == "3":
            break
        else:
            print("Invalid choice. Please select 1-3.")