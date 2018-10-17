'''
HMM based POS tagger <NLP Assignment 4>

Author :
    Shwetank Shrey
Description :
    Implements Viterbi Algorithm on a Hidden Markov Model
    based on a bigram tag / state model using the POS tagged
    section of the BERP corpus.
'''

# Initialise training dataset for the HMM.
training_file = open("training/hmm-training.txt", "r")
all_lines = training_file.readlines()
training_file.close()

list_of_tags = []
tag_bigram = {}
tag_word_likelihood = {}

# Extract required counts from the training dataset.
prev_tag = "<S>"
for line in all_lines:
    if prev_tag == "<E>":
        prev_tag = "<S>"
        continue

    word, tag = line.split()

    if tag == ".":
        tag = "<E>"

    if tag not in list_of_tags:
        list_of_tags.append(tag)

    if prev_tag not in tag_bigram:
        tag_bigram[prev_tag] = {}
    if tag not in tag_bigram[prev_tag]:
        tag_bigram[prev_tag][tag] = 1
    tag_bigram[prev_tag][tag] += 1

    prev_tag = tag
    if tag == "<E>":
        continue

    if tag not in tag_word_likelihood:
        tag_word_likelihood[tag] = {}
    if word not in tag_word_likelihood[tag]:
        tag_word_likelihood[tag][word] = 1
    tag_word_likelihood[tag][word] += 1

# Calculate add one smoothed tag bigram probabilities
for tag in tag_bigram:
    for next_tag in list_of_tags:
        if next_tag not in tag_bigram[tag]:
            tag_bigram[tag][next_tag] = 1
    tag_count = 0
    for next_tag in tag_bigram[tag]:
        tag_count += tag_bigram[tag][next_tag]
    for next_tag in tag_bigram[tag]:
        tag_bigram[tag][next_tag] /= tag_count

# Calculate add one smoothed tag word likelihood probabilities with unknown words dealt using Things seen once
for tag in list_of_tags:
    if tag not in tag_word_likelihood:
        tag_word_likelihood[tag] = {}
for tag in tag_word_likelihood:
    tag_count = 0
    tag_word_likelihood[tag]["<U>"] = 1
    unk_words = []
    for word in tag_word_likelihood[tag]:
        tag_count += tag_word_likelihood[tag][word]
        if tag_word_likelihood[tag][word] == 2 and word != "<U>":
            unk_words.append(word)
            tag_word_likelihood[tag]["<U>"] += 1
    for word in unk_words:
        tag_word_likelihood[tag].pop(word)
    for word in tag_word_likelihood[tag]:
        tag_word_likelihood[tag][word] /= tag_count

list_of_tags.remove("<E>")

# Input sentence for POS tagging
input_file = open("decoding/input.txt", "r")

input_words = []
while True:
    word = input_file.readline()[:-1]
    if word == ".":
        break
    input_words.append(word)
input_length = len(input_words)

expected_pos = {}
state_probability = {}

# Viterbi Algorithm
expected_pos[1] = {}
state_probability[1] = {}
for tag in list_of_tags:
    if input_words[0] in tag_word_likelihood[tag]:
        state_probability[1][tag] = tag_bigram["<S>"][tag] * tag_word_likelihood[tag][input_words[0]]
    else:
        state_probability[1][tag] = tag_bigram["<S>"][tag] * tag_word_likelihood[tag]["<U>"]
    expected_pos[1][tag] = "<S>"

for i in range(2, input_length + 1):
    expected_pos[i] = {}
    state_probability[i] = {}
    for tag in list_of_tags:
        state_probability[i][tag] = 0
        for prev_tag in list_of_tags:
            if input_words[i-1] in tag_word_likelihood[tag]:
                arg = state_probability[i - 1][prev_tag] * tag_bigram[prev_tag][tag] * tag_word_likelihood[tag][input_words[i - 1]]
            else:
                arg = state_probability[i - 1][prev_tag] * tag_bigram[prev_tag][tag] * tag_word_likelihood[tag]["<U>"]
            if arg > state_probability[i][tag]:
                state_probability[i][tag] = arg
                expected_pos[i][tag] = prev_tag

i = input_length + 1
expected_pos[i] = {}
state_probability[i] = {}
state_probability[i]["<E>"] = 0
for prev_tag in list_of_tags:
    arg = state_probability[i - 1][prev_tag] * tag_bigram[prev_tag]["<E>"]
    if arg > state_probability[i]["<E>"]:
        state_probability[i]["<E>"] = arg
        expected_pos[i]["<E>"] = prev_tag

# Output POS tagged using backtrace
input_pos = [""] * (input_length + 2)
prev_tag = "<E>"
for i in range(input_length+1, 0, -1):
    input_pos[i] = expected_pos[i][prev_tag]
    prev_tag = input_pos[i]

for i in range(input_length):
    print(input_words[i] + "\t" + input_pos[i+2])