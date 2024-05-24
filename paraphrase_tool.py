from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import argparse
from tqdm import tqdm
import time

# Load models for English and Indonesian
model_name_en = "tuner007/pegasus_paraphrase"  # You can use any suitable English paraphrase model
model_name_id = "HMehrab/bangla_idiom_paraphrase_v1"  # Example Indonesian model

tokenizer_en = AutoTokenizer.from_pretrained(model_name_en)
model_en = AutoModelForSeq2SeqLM.from_pretrained(model_name_en)

tokenizer_id = AutoTokenizer.from_pretrained(model_name_id)
model_id = AutoModelForSeq2SeqLM.from_pretrained(model_name_id)

# Create pipelines for each language
paraphrase_pipeline_en = pipeline('text2text-generation', model=model_en, tokenizer=tokenizer_en)
paraphrase_pipeline_id = pipeline('text2text-generation', model=model_id, tokenizer=tokenizer_id)

def paraphrase_text(text, language):
    if language == 'en':
        input_text = f"paraphrase: {text} </s>"
        paraphrased = paraphrase_pipeline_en(input_text, max_length=100, num_return_sequences=1)
    elif language == 'id':
        input_text = f"paraphrase: {text} </s>"
        paraphrased = paraphrase_pipeline_id(input_text, max_length=100, num_return_sequences=1)
    else:
        raise ValueError("Unsupported language")
    return paraphrased[0]['generated_text']

def main():
    print("Choose the language for paraphrasing:")
    print("1. English")
    print("2. Indonesian")
    choice = input("Enter the number (1 or 2): ")

    if choice == '1':
        language = 'en'
        text_prompt = "Enter the paragraph to paraphrase: "
    elif choice == '2':
        language = 'id'
        text_prompt = "Masukkan paragraf yang ingin diubah: "
    else:
        print("Invalid choice. Please run the program again and enter 1 or 2.")
        return

    text = input(text_prompt)

    print("Processing your request, please wait...")
    
    # Display the loading bar
    for _ in tqdm(range(10), desc="Paraphrasing", ncols=100, ascii=True):
        time.sleep(0.1)  # Simulate loading time
    
    # Call the paraphrase function
    paraphrased = paraphrase_text(text, language)
    
    if language == 'en':
        print(f"\nOriginal ({language}): {text}")
        print(f"Paraphrased ({language}): {paraphrased}")
    elif language == 'id':
        print(f"\nAsli ({language}): {text}")
        print(f"Parafrase ({language}): {paraphrased}")

if __name__ == "__main__":
    main()
