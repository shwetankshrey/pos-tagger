import pickle

pickle_file = open("training/pickled_model", "rb")
model = pickle.load(pickle_file)
pickle_file.close()
list_of_tags, tag_bigram, tag_word_likelihood = model

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