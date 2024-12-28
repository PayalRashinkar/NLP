import json

# Load HMM probabilities
def load_hmm_probabilities(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

# Greedy Decoding Function
def greedy_decode(sentence, transition_probs, emission_probs):
    prev_tag = 'NNP'  
    predicted_tags = []
    for word in sentence:
        max_prob = 0
        best_tag = None
        for tag, prob in emission_probs.items():
            current_tag = tag[2:-2].split("', '")[0]
            if f"('{current_tag}', '{word}')" in emission_probs:
                emit_prob = emission_probs[f"('{current_tag}', '{word}')"]
                trans_prob = transition_probs.get(f"('{prev_tag}', '{current_tag}')", 0)
                prob = trans_prob * emit_prob
                if prob > max_prob:
                    max_prob = prob
                    best_tag = current_tag
        if best_tag is not None:
            predicted_tags.append(best_tag)
            prev_tag = best_tag
        else:
            predicted_tags.append('NN')
    return predicted_tags

# Read Sentences from Development Data
def read_sentences_from_dev_data(filepath):
    sentences = []
    current_sentence = []
    with open(filepath, 'r') as file:
        for line in file:
            if line.strip():
                parts = line.strip().split('\t')
                if parts[0] == '1' and current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = [parts[1]]
                else:
                    current_sentence.append(parts[1])
        if current_sentence:
            sentences.append(current_sentence)
    return sentences

# Write Predictions to File
def write_predictions_to_file(sentences, predictions, output_file):
    with open(output_file, 'w') as file:
        for sentence, tags in zip(sentences, predictions):
            for i, (word, tag) in enumerate(zip(sentence, tags), 1):
                file.write(f"{i}\t{word}\t{tag}\n")
            file.write("\n")

# Main Execution
if __name__ == "__main__":
    # Load HMM probabilities
    hmm_probs = load_hmm_probabilities('hmm.json')
    transition_probs = hmm_probs['transition']
    emission_probs = hmm_probs['emission']

    # Read sentences from development data
    filepath = 'test'
    sentences = read_sentences_from_dev_data(filepath)

    # Predict POS tags for each sentence and store predictions
    predictions = [greedy_decode(sentence, transition_probs, emission_probs) for sentence in sentences]

    # Write predictions to greedy.out
    write_predictions_to_file(sentences, predictions, 'greedy.out')
