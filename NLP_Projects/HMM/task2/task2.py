from collections import defaultdict, Counter
import json

def learn_hmm_from_file(train_file_path):
    transition_counts = defaultdict(int)
    emission_counts = defaultdict(int)
    state_counts = defaultdict(int)

    # Open the training file
    with open(train_file_path, 'r') as file:
        last_state = None
        for line in file:
            line = line.strip()
            if line:  # Not an empty line
                _, word, state = line.split('\t')
                emission_counts[(state, word)] += 1
                state_counts[state] += 1
                
                if last_state is not None:  # If not the start of a sentence
                    transition_counts[(last_state, state)] += 1
                    
                last_state = state
            else:  # Empty line, indicating the end of a sentence
                last_state = None  # Reset for the next sentence

    # Calculate transition and emission probabilities
    transition_probs = {f"{(prev_state, next_state)}": count / state_counts[prev_state]
                        for (prev_state, next_state), count in transition_counts.items()}
    
    emission_probs = {f"{(state, word)}": count / state_counts[state]
                      for (state, word), count in emission_counts.items()}

    # Prepare the model dictionary
    model = {
        "transition": transition_probs,
        "emission": emission_probs
    }

    # Output the model to a JSON file
    output_file_path = "hmm.json"
    with open(output_file_path, "w") as json_file:
        json.dump(model, json_file, indent=4)

    print(f"Model saved to {output_file_path}. Transition parameters: {len(transition_probs)}, Emission parameters: {len(emission_probs)}.")

# Example usage
train_file_path = "train"
learn_hmm_from_file(train_file_path)

