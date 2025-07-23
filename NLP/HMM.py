from collections import Counter

# === Mini Tagged Corpus ===
tagged_sentences = [
    (["Race", "fast", "clean"], ["VB", "RB", "JJ"]),
    (["Drift", "wild", "corner"], ["VB", "JJ", "NN"]),
    (["Brake", "early", "smoothly"], ["VB", "RB", "RB"])
]

# === Count Start Tags ===
def count_start_tags(tagged_sents):
    start_counts = Counter()
    total = len(tagged_sents)
    for sent in tagged_sents:
        start_tag = sent[1][0]
        start_counts[start_tag] += 1
    return {tag: count / total for tag, count in start_counts.items()}

# === Count Transitions ===
def count_tag_transitions(tagged_sents):
    trans_counts, totals = {}, {}
    for sent in tagged_sents:
        tags = sent[1]
        for i in range(len(tags) - 1):
            prev, curr = tags[i], tags[i+1]
            trans_counts.setdefault(prev, {})
            trans_counts[prev][curr] = trans_counts[prev].get(curr, 0) + 1
            totals[prev] = totals.get(prev, 0) + 1
    for prev in trans_counts:
        for curr in trans_counts[prev]:
            trans_counts[prev][curr] /= totals[prev]
    return trans_counts

# === Count Emissions ===
def count_tag_emissions(tagged_sents):
    emit_counts, totals = {}, {}
    for sent in tagged_sents:
        words, tags = sent[0], sent[1]
        for word, tag in zip(words, tags):
            emit_counts.setdefault(tag, {})
            emit_counts[tag][word] = emit_counts[tag].get(word, 0) + 1
            totals[tag] = totals.get(tag, 0) + 1
    for tag in emit_counts:
        for word in emit_counts[tag]:
            emit_counts[tag][word] /= totals[tag]
    return emit_counts

# === Simple Viterbi Algorithm ===
def simple_viterbi(obs_seq, states, start_prob, trans_prob, emit_prob):
    probability = [{}]
    best_path = {s: [s] for s in states}

    # First observation
    for state in states:
        probability[0][state] = start_prob.get(state, 0) * emit_prob.get(state, {}).get(obs_seq[0], 0.0001)

    # Dynamic programming steps
    for t in range(1, len(obs_seq)):
        probability.append({})
        new_path = {}

        for curr_state in states:
            best_score, best_prev = -1, None
            for prev_state in states:
                score = (
                    probability[t - 1][prev_state] *
                    trans_prob.get(prev_state, {}).get(curr_state, 0) *
                    emit_prob.get(curr_state, {}).get(obs_seq[t], 0.0001)
                )
                if score > best_score:
                    best_score, best_prev = score, prev_state
            probability[t][curr_state] = best_score
            new_path[curr_state] = best_path[best_prev] + [curr_state]
        best_path = new_path

    # Final state
    final_state = max(probability[-1], key=probability[-1].get)
    return best_path[final_state]

# === Generate Probabilities ===
start_prob = count_start_tags(tagged_sentences)
trans_prob = count_tag_transitions(tagged_sentences)
emit_prob = count_tag_emissions(tagged_sentences)
states = ["VB", "RB", "JJ", "NN"]

# === Test Sentence ===
test_sentence = ["Race", "early", "corner"]
predicted_tags = simple_viterbi(test_sentence, states, start_prob, trans_prob, emit_prob)

# === Result ===
print("Sentence:", test_sentence)
print("Predicted Tags:", predicted_tags)
