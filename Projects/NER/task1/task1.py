import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from sklearn.metrics import f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.nn import CrossEntropyLoss

num_epochs = 15
embedding_dim = 100
hidden_dim = 256
num_layers = 1
dropout = 0.33
linear_output_dim = 128

class DatasetLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.dataset_sequences = []
        self.word_set = set()
        self.tag_set = set()
    
    def load_data(self):
        with open(self.file_path, 'r') as dataset_file:
            lines = dataset_file.readlines()
        temp_words, temp_tags = [], []
        for line in lines:
            if line.strip() == "":
                if temp_words and temp_tags:
                    self._update_sets(temp_words, temp_tags)
                temp_words, temp_tags = [], []
            else:
                _, word, tag = line.strip().split()
                temp_words.append(word)
                temp_tags.append(tag)
        
        if temp_words and temp_tags:
            self._update_sets(temp_words, temp_tags)
        return self.dataset_sequences, self.word_set, self.tag_set

    def _update_sets(self, words, tags):
        self.dataset_sequences.append((words, tags))
        self.word_set.update(words)
        self.tag_set.update(tags)

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

def build_vocab_mappings(dataset, distinct_tags, min_freq):
    word_counts = Counter(word for sentence, _ in dataset for word in sentence)
    vocab_words = [word for word, freq in word_counts.items() if freq >= min_freq]

    idx_to_word = {idx + 4: word for idx, word in enumerate(vocab_words)}
    idx_to_word.update({0: '<pad>', 1: '<s>', 2: '</s>', 3: '<unk>'})

    idx_to_tag = {idx + 3: tag for idx, tag in enumerate(distinct_tags)}
    idx_to_tag.update({0: '<pad>', 1: '<s>', 2: '</s>'})

    return {v: k for k, v in idx_to_word.items()}, {v: k for k, v in idx_to_tag.items()}

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

word2idx, tag2idx = build_vocab_mappings(sequences, unique_tags, min_freq=1)
vocab_size = len(word2idx)
num_tags = len(tag2idx)

def sequence_padding(data_batch, word_to_idx, tag_to_idx, padding='<pad>', start='<s>', stop='</s>', unknown='<unk>'):
    max_sequence_length = max(len(sentence) + 2 for sentence, _ in data_batch)
    sequence_word_pads = []
    sequence_tag_pads = []

    for sentence, labels in data_batch:
        padded_sentence = [start] + sentence + [stop]
        padded_sentence = [word_to_idx.get(word, word_to_idx[unknown]) for word in padded_sentence] + [word_to_idx[padding]] * (max_sequence_length - len(padded_sentence))
        sequence_word_pads.append(padded_sentence)
        padded_labels = [start] + labels + [stop]
        padded_labels = [tag_to_idx[label] for label in padded_labels] + [tag_to_idx[padding]] * (max_sequence_length - len(padded_labels))
        sequence_tag_pads.append(padded_labels)

    return torch.tensor(sequence_word_pads, dtype=torch.long), torch.tensor(sequence_tag_pads, dtype=torch.long)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    collate_fn=lambda batch: sequence_padding(batch, word2idx, tag2idx),
    shuffle=True,
)
dev_loader = DataLoader(
    dev_dataset,
    batch_size=8,
    collate_fn=lambda batch: sequence_padding(batch, word2idx, tag2idx),
    shuffle=True,
)

def evaluate_model_performance(model, data_loader, criterion, total_tags):
    model.eval()
    cumulative_loss = 0
    actual_tags = []
    predicted_tags = []
    aggregate_accuracy = 0
    batch_count = 0
    cumulative_loss = 0

    with torch.no_grad():
        for _, batch in enumerate(data_loader):
            texts, labels = batch
            texts = texts.to(device)
            labels = labels.to(device)
            predictions = model(texts)
            predictions = predictions.reshape(-1, total_tags)
            labels = labels.flatten()
            batch_loss = criterion(predictions, labels)
            cumulative_loss += batch_loss.item()
            actual = labels.cpu().numpy()
            predicted = torch.argmax(predictions, axis=1).cpu().numpy()
            actual_tags.extend(actual)
            _, predicted_labels = torch.max(predictions, axis=1)
            predicted_tags.extend(predicted_labels.cpu().numpy())
            valid_predictions = actual != 0
            correct_preds = (predicted[valid_predictions] == actual[valid_predictions]).sum()
            batch_accuracy = correct_preds / len(actual[valid_predictions])       
            aggregate_accuracy += batch_accuracy
            cumulative_loss += batch_loss
            batch_count += 1

    avg_precision, avg_recall, avg_f1, _ = precision_recall_fscore_support(
        actual_tags,
        predicted_tags,
        average='macro',
        zero_division=0
    )

    return (cumulative_loss/batch_count), (aggregate_accuracy/batch_count) * 100, avg_precision * 100, avg_recall * 100, avg_f1 * 100

class BidirectionalLSTMModel(nn.Module):
    def __init__(self, total_vocab_size, output_dimension, embedding_dimension, lstm_hidden_dimension, lstm_layers, dropout_rate):
        super(BidirectionalLSTMModel, self).__init__()
        self.embed = nn.Embedding(total_vocab_size, embedding_dimension)
        self.bi_lstm = nn.LSTM(embedding_dimension, lstm_hidden_dimension, lstm_layers, bidirectional=True, batch_first=True)
        self.dense1 = nn.Linear(lstm_hidden_dimension * 2, output_dimension)
        self.activation = nn.ELU()
        self.regularization = nn.Dropout(dropout_rate)
        self.dense2 = nn.Linear(output_dimension, num_tags)

    def forward(self, text_sequence):
        embedded_text = self.embed(text_sequence)
        lstm_out, _ = self.bi_lstm(embedded_text)
        dense_out = self.dense1(lstm_out)
        activated_out = self.activation(dense_out)
        dropped_out = self.regularization(activated_out)
        final_logits = self.dense2(dropped_out)
        return final_logits

vocab_size = len(word2idx)
num_tags = len(tag2idx)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
final_model = None
highest_f1_score = 0
model = BidirectionalLSTMModel(vocab_size, linear_output_dim, embedding_dim, hidden_dim, num_layers, dropout)
model.to(device)

loss_function = CrossEntropyLoss(ignore_index=tag2idx['<pad>']) 
optimizer = optim.SGD(model.parameters(), lr=0.25, momentum=0.9, weight_decay=0.00005)  # TODO add parameters
patience = 6

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience, factor=0.5, verbose=True)

early_stopping_counter = 0
best_f1_score = -1
clip_value = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    total_samples = 0

    for batch in train_loader:
        word_seqs, tag_seqs = batch
        word_seqs = word_seqs.to(device)
        tag_seqs = tag_seqs.to(device)
        optimizer.zero_grad()
        logits = model(word_seqs)
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
        torch.save(model.state_dict(), "blstm1.pt")
    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, Precision: {val_precision:.4f}, Recall {val_recall:.4f}, F1_score {val_f1_score:.4f}")

loaded_model = BidirectionalLSTMModel(9971, linear_output_dim, embedding_dim, hidden_dim, num_layers, dropout)
saved_state_dict = torch.load("blstm1.pt")
loaded_model.load_state_dict(saved_state_dict)
loaded_model.eval()

def get_outputfile(model, textFile, outputFile, has_tags=True):
    with open(textFile, 'r') as input_file, open(outputFile, 'w') as output_file:
        indexs = []
        words = []
        tags = [] if has_tags else None  
        pad_token='<pad>'
        init_token='<s>'
        eos_token='</s>'
        unk_token='<unk>'
        for line in input_file:
            if not line.strip():
               
                if len(words) > 0:
                    idx2tag = {idx: tag for tag, idx in tag2idx.items()}
                    new_text = " ".join(words)
                    model.eval()
                    tokens = new_text.split()
                    padded_tokens = [init_token] + tokens + [eos_token]
                    indices = [word2idx.get(word, word2idx[unk_token]) for word in padded_tokens]
                    input_tensor = torch.tensor([indices]).to(device)
                    with torch.no_grad():
                        logits = model(input_tensor)              
                    predicted_indices = torch.argmax(logits, dim=-1).squeeze().cpu().numpy()
                    predicted_tags = [idx2tag.get(idx, unk_token) for idx in predicted_indices][1:-1]
                    for i in range(len(indexs)):
                        index = indexs[i]
                        word = words[i]
                        prediction = predicted_tags[i]
                        predictionLine = f"{index} {word} {prediction}\n"
                        output_file.write(predictionLine)                   
                    indexs = []
                    words = []
                    if has_tags: tags = []  
                    output_file.write("\n")
            else: 
                split_line = line.strip().split()
                index, word = split_line[:2]
                indexs.append(index)
                words.append(word)
                if has_tags: tags.append(split_line[2])  

get_outputfile(loaded_model, "data/dev", "dev1.out", has_tags=True)
get_outputfile(loaded_model, "data/test", "test1.out", has_tags=False)
