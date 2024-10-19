Homework 2 - Video Captioning with Attention
Brief Description
This project implements a video captioning system using the S2VT (Sequence to Video to Text) architecture with attention. The system is designed to process video features and generate natural language descriptions for the videos. It includes the following tasks:

Data Preparation:

Vocabulary Creation: A vocabulary is built from the training captions, considering word frequency to filter uncommon words.
Video Caption Dataset: A custom dataset class loads video features and corresponding captions, encoding the captions for training.
Model Design:

S2VT with Attention Model: This model consists of an encoder LSTM for video features, an attention mechanism, and a decoder LSTM to generate the captions. It uses teacher forcing during training.

Training:

Model Training: The model is trained using a cross-entropy loss function, with teacher forcing ratio linearly decreasing across epochs. BLEU scores are used to evaluate the generated captions.

Optimization: Includes gradient clipping to stabilize training, and a learning rate scheduler.
Inference:

Beam Search Decoding: The trained model uses beam search to generate captions, selecting the best sequence of words based on probability scores.


Requirements
Python 3.x
Jupyter Notebook


Required Libraries:
torch
numpy
matplotlib
tqdm
nltk


Installation
Clone the repository:

bash
git clone https://github.com/Mahmoud118/HW2

Change to the project directory:

bash
cd HW2


Launch Jupyter Notebook and run the code:

bash
jupyter notebook


Usage
Data Preparation:

Place the video features (in .npy format) in the folder MLDS_hw2_1_data/training_data/feat/.
Place the captions file (training_label.json) in the project directory.
Training the Model:

The model is trained using the script train_model(), which logs the training loss and BLEU score for each epoch.
Inference:

After training, the model can be used to generate captions for new videos using the beam_search() function.
