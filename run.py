import argparse
from model import loadModel, predict

parser = argparse.ArgumentParser(description="Run sentences through the toxicity classification model.")
# parser.add_argument("--sentence", type=str, required=True, help="The sentence to predict.")
parser.add_argument("--model", type=str, required=True, choices=["roberta", "lstm", "ensemble"], help="The model to use.")
parser.add_argument("--device", type=str, required=False, default='cpu', choices=["cpu", "gpu"], help="The deivce to use to compute.")
parser.add_argument("--checkpoint_dir", type=str, required=False, default='checkpoints/', help="The path to the model checkpoint.")


if __name__ == "__main__":
    args = parser.parse_args()
    
    # Load model
    print(f"Loading {args.model} model from {args.checkpoint_dir}...")

    model, vocab = loadModel(args.model, args.checkpoint_dir, args.device)
    
    while True:
        sentence = input("\nEnter a sentence, or type 'exit' to quit: ").strip()
        if sentence.lower() == "exit":
            print("Exiting. Goodbye!")
            break
        
        if not sentence:
            print("No input received. Please type a sentence.")
            continue

        predicted_class, probabilities = predict(sentence, args.model, model, args.device, vocab)
        
        print(f"{predicted_class} with probability {probabilities:.2f}")
