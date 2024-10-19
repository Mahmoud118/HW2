import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import random

# Hyperparameters
BATCH_SIZE = 36  
EPOCHS = 200  
LEARNING_RATE = 0.0001
HIDDEN_SIZE = 1024
NUM_LAYERS = 5
MAX_SEQ_LENGTH = 30
TEACHER_FORCING_RATIO_START = 0.7 
TEACHER_FORCING_RATIO_END = 0.2
BEAM_SIZE = 5  

# Special tokens
PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2
UNK_IDX = 3

device = torch.device('cuda')

class VideoCaptionDataset(Dataset):
    def __init__(self, video_folder, captions_file, vocab):
        self.video_folder = video_folder
        self.captions_data = json.load(open(captions_file))
        self.vocab = vocab

    def __len__(self):
        return len(self.captions_data)

    def __getitem__(self, idx):
        video_id = self.captions_data[idx]['id']
        video_features = np.load(os.path.join(self.video_folder, f"{video_id}.npy"))
        video_features = video_features.reshape(video_features.shape[0], -1)
        caption = np.random.choice(self.captions_data[idx]['caption'])
        caption_tensor = self.encode_caption(caption)
        return torch.FloatTensor(video_features), caption_tensor

    def encode_caption(self, caption):
        tokens = caption.split()
        encoded = [self.vocab.get(token, UNK_IDX) for token in tokens]
        encoded = [BOS_IDX] + encoded + [EOS_IDX]
        encoded = encoded[:MAX_SEQ_LENGTH]
        while len(encoded) < MAX_SEQ_LENGTH:
            encoded.append(PAD_IDX)
        return torch.LongTensor(encoded)

def create_vocab(captions_file, min_count=4):
    word_count = {}
    captions_data = json.load(open(captions_file))
    
    for item in captions_data:
        for caption in item['caption']:
            for word in caption.split():
                word_count[word] = word_count.get(word, 0) + 1
    
    vocab = {
        '<PAD>': PAD_IDX,
        '<BOS>': BOS_IDX,
        '<EOS>': EOS_IDX,
        '<UNK>': UNK_IDX
    }
    idx = 4
    for word, count in word_count.items():
        if count > 3:
            vocab[word] = idx
            idx += 1
    
    return vocab

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(1)
        h = hidden.repeat(seq_len, 1, 1).transpose(0, 1)
        energy = torch.tanh(self.attn(torch.cat((h, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)


class S2VTWithAttention(nn.Module):
    def __init__(self, vocab_size, feature_size, hidden_size, num_layers):
        super(S2VTWithAttention, self).__init__()
        self.encoder_lstm = nn.LSTM(input_size=feature_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=hidden_size, padding_idx=PAD_IDX)
        self.attention = Attention(hidden_size)
        self.decoder_lstm = nn.LSTM(input_size=hidden_size * 2, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, video_features, captions, teacher_forcing_ratio=0.5):
        encoder_outputs, (hidden, cell) = self.encoder_lstm(video_features)
        batch_size = captions.size(0)
        max_length = captions.size(1) - 1 
        vocab_size = self.fc.out_features

        outputs = torch.zeros(batch_size, max_length, vocab_size).to(captions.device)
        input = captions[:, 0]

        for t in range(max_length):
            embedded = self.embedding(input).unsqueeze(1)
            attn_weights = self.attention(hidden[-1], encoder_outputs)
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
            lstm_input = torch.cat((embedded, context), dim=2)
            output, (hidden, cell) = self.decoder_lstm(lstm_input, (hidden, cell))
            output = self.fc(output.squeeze(1))
            outputs[:, t] = output

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = captions[:, t+1] if teacher_force else top1

        return outputs

    def encode(self, video_features):
        return self.encoder_lstm(video_features)

    def decode_step(self, input, hidden, cell, encoder_outputs):
        embedded = self.embedding(input).unsqueeze(1)
        attn_weights = self.attention(hidden[-1], encoder_outputs)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)
        lstm_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.decoder_lstm(lstm_input, (hidden, cell))
        output = self.fc(output.squeeze(1))
        return output, hidden, cell

def calculate_bleu(candidate, reference):
    smooth_fn = SmoothingFunction().method1
    candidate_tokens = [str(idx) for idx in candidate]
    reference_tokens = [[str(idx) for idx in reference]]
    return sentence_bleu(reference_tokens, candidate_tokens, weights=(1,), smoothing_function=smooth_fn)



def train_model(model, dataloader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        total_bleu = 0
        teacher_forcing_ratio = TEACHER_FORCING_RATIO_START - (TEACHER_FORCING_RATIO_START - TEACHER_FORCING_RATIO_END) * (epoch / epochs)

        for i, (video_features, captions) in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()

            video_features = video_features.to(device)
            captions = captions.to(device)

            outputs = model(video_features, captions, teacher_forcing_ratio)

            loss = criterion(outputs.view(-1, outputs.size(-1)), captions[:, 1:].contiguous().view(-1))
            loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()

            pred = outputs[0].argmax(dim=1)
            target = captions[0, 1:]
            bleu_score = calculate_bleu(pred, target)
            total_bleu += bleu_score

            if i % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}, BLEU@1: {bleu_score:.4f}')
                print("Sample prediction:")
                print("Pred:", ' '.join([str(idx.item()) for idx in pred]))
                print("Target:", ' '.join([str(idx.item()) for idx in target]))
                print()

        avg_loss = total_loss / len(dataloader)
        avg_bleu = total_bleu / len(dataloader)
        print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}, Average BLEU@1: {avg_bleu:.4f}')


        scheduler.step()  
        
        avg_loss = total_loss / len(dataloader)
        avg_bleu = total_bleu / len(dataloader)
        print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}, Average BLEU@1: {avg_bleu:.4f}, Teacher Forcing Ratio: {teacher_forcing_ratio:.4f}')

        teacher_forcing_ratio = max(TEACHER_FORCING_RATIO_END, TEACHER_FORCING_RATIO_START * (1 - (epoch + 1) / epochs))


def beam_search(model, video_features, vocab, beam_size=3, max_length=MAX_SEQ_LENGTH):
    model.eval()
    with torch.no_grad():
        video_features = video_features.to(next(model.parameters()).device) 
        
        encoder_outputs, (hidden, cell) = model.encode(video_features.unsqueeze(0))
        
        beam = [(torch.LongTensor([BOS_IDX]).to(video_features.device), 0, hidden, cell)]
        
        for _ in range(max_length):
            candidates = []
            
            for seq, score, h, c in beam:
                if seq[-1].item() == EOS_IDX:
                    candidates.append((seq, score, h, c))
                    continue
                
                input = seq[-1].unsqueeze(0).to(video_features.device) 
                output, new_h, new_c = model.decode_step(input, h, c, encoder_outputs)
                
                topk_probs, topk_idx = output.topk(beam_size)
                
                for prob, idx in zip(topk_probs[0], topk_idx[0]):
                    idx = idx.unsqueeze(0).to(seq.device)  
                    new_seq = torch.cat([seq, idx]) 
                    new_score = score - torch.log(prob).item()
                    candidates.append((new_seq, new_score, new_h, new_c))
            
            beam = sorted(candidates, key=lambda x: x[1])[:beam_size]
            
            if all(seq[-1].item() == EOS_IDX for seq, _, _, _ in beam):
                break
    
    # Return the sequence with the best score
    best_seq = beam[0][0]
    
    reverse_vocab = {idx: word for word, idx in vocab.items()}
    return ' '.join([reverse_vocab.get(idx.item(), '<UNK>') for idx in best_seq if idx.item() not in [BOS_IDX, EOS_IDX, PAD_IDX]])


if __name__ == "__main__":
    video_folder = 'MLDS_hw2_1_data/training_data/feat/'
    captions_file = 'MLDS_hw2_1_data/training_label.json'
    model_save_dir = 'saved_models/'  
    os.makedirs(model_save_dir, exist_ok=True) 

    vocab = create_vocab(captions_file)
    print(f"Vocabulary size: {len(vocab)}")

    dataset = VideoCaptionDataset(video_folder, captions_file, vocab)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    feature_size = next(iter(dataloader))[0].shape[-1]
    print(f"Feature size: {feature_size}")

    model = S2VTWithAttention(len(vocab), feature_size, HIDDEN_SIZE, NUM_LAYERS).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    train_model(model, dataloader, criterion, optimizer, EPOCHS)

    video_id = '_JVxurtGIhI_32_42.avi'
    video_features = np.load(os.path.join(video_folder, f"{video_id}.npy"))
    video_features = torch.FloatTensor(video_features.reshape(video_features.shape[0], -1))
    generated_caption = beam_search(model, video_features, vocab, beam_size=BEAM_SIZE)
    print("Generated Caption:", generated_caption)

    model_save_path = os.path.join(model_save_dir, 'video_captioning_model.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')

    # Testing the model
    test_video_folder = 'MLDS_hw2_1_data/testing_data/feat/'
    test_captions_file = 'MLDS_hw2_1_data/testing_label.json'
    
    test_dataset = VideoCaptionDataset(test_video_folder, test_captions_file, vocab)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Evaluate the model on the test dataset
    model.eval()
    all_bleu_scores = []

    with torch.no_grad():
        for video_features, captions in tqdm(test_dataloader):
            video_features = video_features.to(device)

            for idx in range(video_features.size(0)):
                generated_caption = beam_search(model, video_features[idx], vocab, beam_size=BEAM_SIZE)
                reference_captions = [captions[idx][1:-1]]  # Get the reference caption without <BOS> and <EOS>

                # Calculate BLEU score
                bleu_score = calculate_bleu(generated_caption, reference_captions)
                all_bleu_scores.append(bleu_score)

    avg_bleu = sum(all_bleu_scores) / len(all_bleu_scores)
    print(f'Average BLEU Score on Test Data: {avg_bleu:.4f}')