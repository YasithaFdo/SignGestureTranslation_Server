from transformers import BartForConditionalGeneration, BartTokenizer

bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

def generate(predicted_words_list):
    # Load BART model and tokenizer

    refined_sentences = []

    for predicted_words in predicted_words_list:
        # Join the predicted words to form a masked sentence
        masked_sentence = " <mask> ".join(predicted_words)

        # Tokenize the masked sentence
        tokenized_sent = tokenizer(masked_sentence, return_tensors='pt')

        # Generate refined output using BART
        generated_encoded = bart_model.generate(
            tokenized_sent['input_ids'],
            max_length=50,  # Allow room for a full sentence
            num_beams=5,  # Use beam search for better predictions
            early_stopping=True
        )

        # Decode the generated sentence
        refined_sentence = tokenizer.decode(generated_encoded[0], skip_special_tokens=True)
        refined_sentences.append(refined_sentence)

    return refined_sentences


# Example usage
predicted_words_list = [
    ["I", "Want", "Go", "School", "I", "Ill"],
    ["she", "go", "market", "buy", "fruit"],
    ["Mother", "eat", "food","she", "hungry"],
    ["Dog", "bark", "person"],
    ["He", "not", "play", "football", "leg", "broken"]
]

output_sentences = generate(predicted_words_list)
for sentence in output_sentences:
    print(sentence)
