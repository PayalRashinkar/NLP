from collections import Counter

# Load training data
def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        words = [line.split('\t')[1] for line in file.read().splitlines() if line]
    return words

# Create vocabulary
def create_vocabulary(words, threshold=3):
    # Count word occurrences
    word_counts = Counter(words)
    # Initialize vocabulary with '<unk>' to ensure it's added first
    vocab = {}
    unk_count = 0
    for word, count in word_counts.items():
        if count < threshold:
            unk_count += count
        else:
            vocab[word] = count
    # Add '<unk>' with its total count at the beginning of the vocabulary
    sorted_vocab = {'<unk>': unk_count}
    # Sort the rest of the vocabulary by occurrences in descending order and update the dictionary
    sorted_vocab.update(dict(sorted(vocab.items(), key=lambda item: item[1], reverse=True)))
    return sorted_vocab

# Save vocabulary to a file
def save_vocabulary(vocab, filepath):
    with open(filepath, 'w', encoding='utf-8') as file:
        for index, (word, count) in enumerate(vocab.items()):
            file.write(f"{word}\t{index}\t{count}\n")

# Main execution
if __name__ == "__main__":
    training_data_path = "train"  
    vocab_file_path = "vocab.txt"
    words = load_data(training_data_path)
    vocab = create_vocabulary(words)
    save_vocabulary(vocab, vocab_file_path)
    print(f"Vocabulary created and saved to {vocab_file_path}")
