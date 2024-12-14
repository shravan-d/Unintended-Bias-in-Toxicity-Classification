import argparse
from model import loadModel, predict
import torch

parser = argparse.ArgumentParser(description="Run sentences through the toxicity classification model.")
parser.add_argument("--model", type=str, required=True, choices=["roberta", "lstm", "ensemble"], help="The model to use.")
parser.add_argument("--checkpoint_dir", type=str, required=False, default='checkpoints/', help="The path to the model checkpoint.")


if __name__ == "__main__":
    args = parser.parse_args()
    
    model_name = args.model.lower()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading {args.model} model from {args.checkpoint_dir}...")
    model, vocab, tokenizer = loadModel(model_name, args.checkpoint_dir, device)
    
    while True:
        sentence = input("\nEnter a sentence, or type 'exit' to quit: ").strip()
        if sentence.lower() == "exit":
            print("Exiting. Goodbye!")
            break
        
        if not sentence:
            print("No input received. Please type a sentence.")
            continue

        predicted_class, probabilities = predict(sentence, model_name, model, device, vocab, tokenizer)
        
        print(f"{predicted_class} with probability {probabilities:.2f}")
