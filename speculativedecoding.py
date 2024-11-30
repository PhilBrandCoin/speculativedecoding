.

from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the draft (smaller) and target (larger) models
draft_model_name = "distilgpt2"  # Smaller, faster model
target_model_name = "gpt2"       # Larger, more accurate model

draft_tokenizer = AutoTokenizer.from_pretrained(draft_model_name)
draft_model = AutoModelForCausalLM.from_pretrained(draft_model_name)

target_tokenizer = AutoTokenizer.from_pretrained(target_model_name)
target_model = AutoModelForCausalLM.from_pretrained(target_model_name)

def speculative_decode(prompt, max_length=50, threshold=0.9):
    """
    Performs speculative decoding using a draft model and a target model.

    Args:
        prompt (str): The input text to the model.
        max_length (int): The maximum length of the generated sequence.
        threshold (float): The confidence threshold for the target model to accept tokens.

    Returns:
        str: The generated text sequence.
    """
    # Step 1: Generate tokens using the draft model
    draft_inputs = draft_tokenizer(prompt, return_tensors="pt")  # Tokenize the prompt
    draft_outputs = draft_model.generate(draft_inputs.input_ids, max_length=max_length)  # Generate draft tokens
    draft_tokens = draft_tokenizer.decode(draft_outputs[0], skip_special_tokens=True)  # Decode draft tokens to text

    # Step 2: Verify tokens using the target model
    target_inputs = target_tokenizer(draft_tokens, return_tensors="pt")  # Tokenize the draft output for the target model
    target_logits = target_model(**target_inputs).logits  # Get logits (un-normalized probabilities) from the target model

    # Step 3: Calculate confidence scores for each token
    probs = target_logits.softmax(dim=-1)  # Convert logits to probabilities
    confidence = probs.max(dim=-1).values.mean().item()  # Calculate the average confidence score across all tokens

    # Step 4: Decide whether to accept or refine tokens
    if confidence >= threshold:  # If confidence exceeds the threshold
        return draft_tokens  # Accept the draft tokens
    else:  # Otherwise, refine the sequence using the target model
        refined_outputs = target_model.generate(target_inputs.input_ids, max_length=max_length)
        return target_tokenizer.decode(refined_outputs[0], skip_special_tokens=True)  # Return the refined sequence

# Example Usage
prompt = "The future of AI is"
output = speculative_decode(prompt, max_length=20)
print(output)
