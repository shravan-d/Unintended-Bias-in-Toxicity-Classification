# Unintended-Bias-in-Toxicity-Classification
Detecting toxic comments while minimizing unintended model bias.

## Installation

1. Clone the repository:
```
git clone https://github.com/shravan-d/Unintended-Bias-in-Toxicity-Classification.git
cd toxicity-detection-bias-mitigation
```
2. Install dependencies
```
pip install -r requirements.txt
```

3. Ensure you have the model checkpoints in a checkpoints directory in the root of the project. If not, download the files from this link: [Drive](https://drive.google.com/drive/folders/1YbqEKaYfCv2DSQ-axub6Q3Sia-UcWOqZ), create a directory called checkpoints in the root, and place the files there.

4. Download the glove embeddings from this link [Drive](https://drive.google.com/drive/folders/1YbqEKaYfCv2DSQ-axub6Q3Sia-UcWOqZ), and place them in data/glove.6B

5. To classify sentences, run the script and input your sentences one by one.
```
python run.py --model_name ensemble
```
