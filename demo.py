import argparse
from model import loadModel, predict
import torch

parser = argparse.ArgumentParser(description="Run pre-determined sentences through the toxicity classification model.")
parser.add_argument("--model", type=str, required=True, choices=["roberta", "lstm", "ensemble"], help="The model to use.")
parser.add_argument("--checkpoint_dir", type=str, required=False, default='checkpoints/', help="The path to the model checkpoint.")


if __name__ == "__main__":
    args = parser.parse_args()
    
    model_name = args.model.lower()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading {args.model} model from {args.checkpoint_dir}...")
    model, vocab, tokenizer = loadModel(model_name, args.checkpoint_dir, device)
    
    sentences = ['Hi, My name is Shravan', 'White people are not all racist', 'The LGBTQ commnity should stand strong']
    for sentence in sentences:
        print(sentence)
        predicted_class, probabilities = predict(sentence, model_name, model, device, vocab, tokenizer)
        
        print(f"Toxicity probability {probabilities:.2f}")
