import sklearn_crfsuite
import re


def word_shape(word):
    # Replace character types with markers
    shape = re.sub(r'[A-Z]', 'X', word)
    shape = re.sub(r'[a-z]', 'x', shape)
    shape = re.sub(r'[0-9]', 'd', shape)
    # Replace non-word characters like '-' or '.'
    shape = re.sub(r'\W', '-', shape)
    return shape

# Step 1: Feature extraction for each word


def extract_features(sentence, index):
    word = sentence[index]
    return {
        'word.lower': word.lower(),
        'is_upper': word.isupper(),
        'suffix_3': word[-3:],
        'word_shape': word_shape(word),
        'prev_word': sentence[index - 1] if index > 0 else "<START>",
        'next_word': sentence[index + 1] if index < len(sentence) - 1 else "<END>"
    }


# Step 2: Prepare training data
train_sents = [
    (["Race", "fast", "clean"], ["VB", "RB", "JJ"]),
    (["Brake", "early", "tight"], ["VB", "RB", "JJ"]),
    (["Drift", "wild", "corner"], ["VB", "JJ", "NN"])
]
X_train = [[extract_features(sent, i) for i in range(len(sent))]
           for sent, _ in train_sents]
y_train = [tags for _, tags in train_sents]

# Step 3: Train CRF
crf = sklearn_crfsuite.CRF(algorithm='lbfgs', max_iterations=100)
crf.fit(X_train, y_train)

# Step 4: Predict tags
test_sent = ["Race", "fast", "clean"]
X_test = [extract_features(test_sent, i) for i in range(len(test_sent))]
predicted_tags = crf.predict_single(X_test)
print("CRF Predicted Tags:", predicted_tags)
