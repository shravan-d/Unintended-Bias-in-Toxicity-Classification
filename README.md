# Unintended-Bias-in-Toxicity-Classification
Detecting toxic comments while minimizing unintended model bias.

## Installation

1. Clone the repository:
```
git clone https://github.com/shravan-d/Unintended-Bias-in-Toxicity-Classification.git
cd Unintended-Bias-in-Toxicity-Classification
```
2. Install dependencies
```
pip install -r requirements.txt
```

3. Download the required embeddings and model checkpoints from the following link: [Drive](https://drive.google.com/drive/folders/1YbqEKaYfCv2DSQ-axub6Q3Sia-UcWOqZ).

4. Move the 'LSTM.pth', 'model.safetensors', and 'config.json' into a folder called 'checkpoints' in the root.

5. Move the 'glove.6B.100d.txt' file into data/glove.6B.

6. To classify sentences, run the script and input your sentences one by one.
```
python demo.py --model_name ensemble
```

7. To classify your won sentences, run the script and input your sentences one by one.
```
python run.py --model_name ensemble
```
