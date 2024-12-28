import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
from collections import Counter
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingLR
glove_file = "./glove.6B.100d.txt"

class DatasetLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.dataset_sequences = []
        self.word_set = set()
        self.tag_set = set()
    
    def load_data(self):
        with open(self.file_path, 'r') as file:
            lines = file.readlines()

        temp_words, temp_tags = [], []
        for line in lines:
            if line.strip() == "":
                if temp_words and temp_tags:
                    self.dataset_sequences.append((temp_words, temp_tags))
                    self.word_set.update(temp_words)
                    self.tag_set.update(temp_tags)
                temp_words, temp_tags = [], []
            else:
                _, word, tag = line.strip().split()
                temp_words.append(word)
                temp_tags.append(tag)

        if temp_words and temp_tags:
            self.dataset_sequences.append((temp_words, temp_tags))
            self.word_set.update(temp_words)
            self.tag_set.update(temp_tags)
        
        return self.dataset_sequences, self.word_set, self.tag_set

class EnhancedDataLoader(Dataset):
    def __init__(self, dataset, modification=None):
        self.dataset = dataset
        self.modification = modification

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.numpy()

        data_point = self.dataset[index]

        if self.modification:
            data_point = self.modification(data_point)

        return data_point

file_path = "data/train"
loader = DatasetLoader(file_path)
sequences, unique_words, unique_tags = loader.load_data()
tokenized_data = [([word for word in words], [tag for tag in tags]) for words, tags in sequences]
train_dataset = EnhancedDataLoader(tokenized_data)

file_path = "data/dev"
loader = DatasetLoader(file_path)
sequences, unique_words, unique_tags = loader.load_data()
tokenized_data = [([word for word in words], [tag for tag in tags]) for words, tags in sequences]
dev_dataset = EnhancedDataLoader(tokenized_data)

def build_vocab_mappings(dataset, distinct_tags, min_freq):
    word_counts = Counter(word.lower() for sentence, _ in dataset for word in sentence)
    vocab_words = [word for word, freq in word_counts.items() if freq >= min_freq]

    idx_to_word = {idx + 4: word for idx, word in enumerate(vocab_words)}
    idx_to_word.update({0: '<pad>', 1: '<s>', 2: '</s>', 3: '<unk>'})

    idx_to_tag = {idx + 3: tag for idx, tag in enumerate(distinct_tags)}
    idx_to_tag.update({0: '<pad>', 1: '<s>', 2: '</s>'})

    word2idx = {word: idx for idx, word in idx_to_word.items()}
    tag2idx = {tag: idx for idx, tag in idx_to_tag.items()}

    return word2idx, tag2idx

word2idx, tag2idx = build_vocab_mappings(sequences, unique_tags, 1)

def pad_sequences(batch, word2idx, tag2idx, pad_token='<pad>', init_token='<s>', eos_token='</s>', unk_token='<unk>'):
    max_len = max([len(seq) + 2 for seq, _ in batch])  # Add 2 to account for <s> and </s> tokens

    padded_word_seqs = []
    padded_upper_seqs = []
    padded_tag_seqs = []

    for words, tags in batch:
        lower_words = [word.lower() for word in words]

        padded_words = [init_token] + lower_words + [eos_token]
        padded_words = [word2idx.get(word, word2idx[unk_token]) for word in padded_words] + [word2idx[pad_token]] * (max_len - len(padded_words))
        padded_word_seqs.append(padded_words)

        padded_uppers = [0] + [int(word[0].isupper()) for word in words] + [0] + [0] * (max_len - len(words) - 2)
        padded_upper_seqs.append(padded_uppers)

        padded_tags = [init_token] + tags + [eos_token]
        padded_tags = [tag2idx[tag] for tag in padded_tags] + [tag2idx[pad_token]] * (max_len - len(padded_tags))
        padded_tag_seqs.append(padded_tags)

    return torch.tensor(padded_word_seqs), torch.tensor(padded_upper_seqs), torch.tensor(padded_tag_seqs)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    collate_fn=lambda batch: pad_sequences(batch, word2idx, tag2idx),
    shuffle=True,
)
dev_loader = DataLoader(
    dev_dataset,
    batch_size=8,
    collate_fn=lambda batch: pad_sequences(batch, word2idx, tag2idx),
    shuffle=True,
)

def load_glove_embeddings(path, word2idx, embedding_dim=100):
    with open(path, 'r', encoding='utf-8') as f:
        embeddings = np.zeros((len(word2idx), embedding_dim))
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            idx = word2idx.get(word)
            if idx is not None:
                embeddings[idx] = vector
    return embeddings

vocab_size = len(word2idx)
num_tags = len(tag2idx)
embedding_dim = 100
hidden_dim = 256
num_layers = 1
dropout = 0.33
linear_output_dim = 128

class BiLSTM(nn.Module):
    def __init__(self, vocab_size, linear_output_dim, embedding_dim, hidden_dim, num_layers, dropout, pretrained_embeddings=None):
        super(BiLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embeddings is not None:
            self.embedding.weight = nn.Parameter(torch.tensor(pretrained_embeddings, dtype=torch.float32))
            self.embedding.weight.requires_grad = False  # Optionally make embeddings non-trainable
        
        self.upper_embedding = nn.Embedding(2, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim * 2, hidden_dim, num_layers, bidirectional=True, batch_first=True)
        self.linear1 = nn.Linear(hidden_dim * 2, linear_output_dim)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(linear_output_dim, num_tags)

    def forward(self, x, upper_x):
        x = self.embedding(x)
        upper_x = self.upper_embedding(upper_x)
        x = torch.cat([x, upper_x], dim=-1)
        x, _ = self.lstm(x)
        x = self.linear1(x)
        x = self.elu(x)
        x = self.dropout(x)
        logits = self.linear2(x)

        return logits

pretrained_embeddings = load_glove_embeddings(glove_file, word2idx, embedding_dim)

print(tag2idx)

weight_list = [0,0,0,1,1,1,0.7,1,1,1,1,1]

for i,w in tag2idx.items():
    print(i, weight_list[w])
print(len(weight_list))
weight_tensor = torch.tensor(weight_list, dtype=torch.float)

def evaluate_model_performance(model, data_loader, criterion, total_tags):
    model.eval()

    cumulative_loss = 0
    actual_tags = []
    predicted_tags = []

    aggregate_accuracy = 0
    batch_count = 0

    with torch.no_grad():
        for batch in data_loader:
            word_seqs, upper_seqs, tag_seqs = batch
            word_seqs = word_seqs.to(device)
            upper_seqs = upper_seqs.to(device)
            tag_seqs = tag_seqs.to(device)

            predictions = model(word_seqs, upper_seqs)
            predictions = predictions.view(-1, total_tags)
            tag_seqs = tag_seqs.view(-1)

            batch_loss = criterion(predictions, tag_seqs)
            cumulative_loss += batch_loss.item()

            actual = tag_seqs.cpu().numpy()
            predicted = torch.argmax(predictions, dim=1).cpu().numpy()
            actual_tags.extend(actual)
            predicted_tags.extend(predicted)

            valid_predictions = actual != 0
            correct_preds = (predicted[valid_predictions] == actual[valid_predictions]).sum()
            batch_accuracy = correct_preds / len(actual[valid_predictions]) if len(actual[valid_predictions]) > 0 else 0
            
            aggregate_accuracy += batch_accuracy
            batch_count += 1

    avg_precision, avg_recall, avg_f1, _ = precision_recall_fscore_support(
        actual_tags,
        predicted_tags,
        average='macro',
        zero_division=0
    )

    return (cumulative_loss/batch_count), (aggregate_accuracy/batch_count) * 100, avg_precision * 100, avg_recall * 100, avg_f1 * 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

final_model = None
highest_f1_score = 0

model = BiLSTM(vocab_size, linear_output_dim, embedding_dim, hidden_dim, num_layers, dropout, pretrained_embeddings)
model.to(device)

num_epochs = 20

loss_function = CrossEntropyLoss(ignore_index=tag2idx['<pad>'], weight=weight_tensor)
optimizer = optim.SGD(model.parameters(), lr=0.3, momentum=0.9, weight_decay=0.00005) 

patience = 5

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience, factor=0.5, verbose=True)

early_stopping_counter = 0
best_f1_score = -1
clip_value = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_samples = 0

    for batch in train_loader:
        word_seqs, upper_seqs, tag_seqs = batch
        word_seqs = word_seqs.to(device)
        upper_seqs = upper_seqs.to(device)
        tag_seqs = tag_seqs.to(device)

        optimizer.zero_grad()

        logits = model(word_seqs, upper_seqs)
        logits = logits.view(-1, num_tags)
        tag_seqs = tag_seqs.view(-1)

        loss = loss_function(logits, tag_seqs)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        optimizer.step()

        total_loss += loss.item() * word_seqs.size(0)
        total_samples += word_seqs.size(0)

    avg_train_loss = total_loss / total_samples

    

    val_loss, val_accuracy, val_precision, val_recall, val_f1_score = evaluate_model_performance(model, dev_loader, loss_function, num_tags)

    scheduler.step(val_loss)

    if val_f1_score > best_f1_score:
        best_f1_score = val_f1_score
        final_model = model

    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall {val_recall:.4f}, F1_score {val_f1_score:.4f}")

torch.save(model.state_dict(), "blstm2.pt")

loaded_model = BiLSTM(vocab_size, linear_output_dim, embedding_dim, hidden_dim, num_layers, dropout, pretrained_embeddings)

saved_state_dict = torch.load("blstm2.pt")
loaded_model.load_state_dict(saved_state_dict)
loaded_model.eval()

def createFile(model, textFile, outputFile):
    with open(textFile, 'r') as input_file, open(outputFile, 'w') as output_file:
        indexs = []
        words = []
        tags = []
        pad_token='<pad>'
        init_token='<s>'
        eos_token='</s>'
        unk_token='<unk>'
        for line in input_file:
            if not line.strip():
                if words:  
                    idx2tag = {idx: tag for tag, idx in tag2idx.items()}
                    new_text = " ".join(words)
                    model.eval()
                    tokens = new_text.split()
                    lower_tokens = new_text.lower().split()
                    padded_tokens = [init_token] + lower_tokens + [eos_token]
                    tokenized_input = [word2idx.get(word, word2idx[unk_token]) for word in padded_tokens]
                    upper_input = [0] + [int(token[0].isupper()) for token in tokens] + [0]
                    input_tensor = torch.tensor([tokenized_input]).to(device)
                    upper_tensor = torch.tensor([upper_input]).to(device)
                    with torch.no_grad():
                        logits = model(input_tensor, upper_tensor)
                    predicted_indices = torch.argmax(logits, dim=-1).squeeze().cpu().numpy()
                    predicted_tags = [idx2tag[idx] for idx in predicted_indices][1:-1]

                    for index, word, prediction in zip(indexs, words, predicted_tags):
                        predictionLine = f"{index} {word} {prediction}\n"
                        output_file.write(predictionLine)

                    indexs, words, tags = [], [], [] 
                    output_file.write("\n")
            else:
                index, word, tag = line.strip().split()
                indexs.append(index)
                words.append(word)
                tags.append(tag)

createFile(loaded_model, "data/dev", "dev2.out")
createFile(loaded_model, "data/dev", "test2.out")

