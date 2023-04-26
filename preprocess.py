import os, pickle, sys, string, unicodedata
import numpy as np
from tqdm.auto import tqdm
from collections import Counter
import random

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

def clean_str(string):
    string = string.strip().lower()
    string = string.replace(".", "")
    string = string.replace("\'s", "")
    string = string.replace("\'", "")
    string = string.replace("\"", "")
    string = string.replace(",", "")
    string = string.replace("!", "")
    string = string.replace("?", "")
    string = string.replace("\n", " ")
    string = string.replace("\\", " ")
    string = string.replace("//", " ")
    string = string.replace("/", " ")
    string = string.replace("<", " ")
    string = string.replace(">", " ")
    string = string.replace("=", " ")
    string = string.replace("_", " ")
    string = string.replace("+", " ")
    string = string.replace("  n", "")
    string = string.replace("-", " ")
    string = string.replace(")", " ")
    string = string.replace("(", " ")
    string = string.replace("[", " ")
    string = string.replace("]", " ")
    string = string.replace("{", " ")
    string = string.replace("}", " ")
    string = string.replace(";", " ")
    string = string.replace(":", " ")
    string = string.replace("@", " ")
    string = string.replace("#", " ")
    string = string.replace("$", " ")
    string = string.replace("%", " ")
    string = string.replace("^", " ")
    string = string.replace("&", " ")
    string = string.replace("*", " ")
    string = string.replace("  ", " ")
    return string.strip()

def freq(text, threshold=5):
    tmp = text.split()

    counts = Counter(tmp)
    output = []
    for w in tqdm(tmp, desc="Discarding rare words"):
        if w.isalpha():
            if counts[w] >= threshold:
                output.append(w)

    return output

def count_freq(text):
    tmp = text.split()
    counts = Counter(tmp)

    return counts

def preprocess(corpus):
    corpus_cnt = Counter(corpus)
    max_cnt = max(corpus_cnt.values())
    corpus_cnt["<pad>"] = max_cnt + 1
    cnts = sorted(corpus_cnt, key = lambda x: -corpus_cnt[x])
    
    output = []
    w2id, id2w = {}, {}

    for word in tqdm(cnts, desc="Making Dictionary"):
        idx = len(w2id)
        w2id[word] = idx
        id2w[idx] = word

    for word in tqdm(corpus, desc="Mapping words with indices"):
        output.append(w2id[word])

    return np.array(output), w2id, id2w, corpus_cnt

def subsampling(corpus, counter, threshold=1e-5):
    subsample_scores = {}
    total_cnt = sum(counter.values()) - counter["<pad>"]
    for word, cnt in tqdm(counter.items(), desc="Calculating subsampling scores"):
        score = 1 - np.sqrt(threshold / (cnt/total_cnt))
        subsample_scores[word] = score
    subsample_scores["<pad>"] = 10.

    return subsample_scores


def subsample_create_contexts_target(corpus, window_size, counter, threshold=1e-5):
    subsample_scores = subsampling(corpus, counter, threshold)
    targets, contexts = [], []

    for idx in tqdm(range(window_size, len(corpus)-window_size), desc="Creating contexts and targets"):
        draw = np.random.uniform()
        if draw > subsample_scores[id2w[corpus[idx]]]:
            targets.append(corpus[idx])
            cs = []
            dynamic_window = random.randint(1, window_size)
            for t in range(-window_size, window_size+1):
                if (t < -dynamic_window) or (t > dynamic_window):
                    cs.append(0)
                else:
                    if t != 0:
                        cs.append(corpus[idx+t])

            contexts.append(cs)
    
    return np.array(contexts), np.array(targets)

def create_contexts_target(corpus, window_size=5):
    targets = corpus[window_size:len(corpus)-window_size]
    contexts = []

    for idx in tqdm(range(window_size, len(corpus)-window_size), desc="Creating contexts and targets"):
        cs = []
        dynamic_window = random.randint(1, window_size)
        for t in range(-window_size, window_size+1):
            if (t < -dynamic_window) or (t > dynamic_window):
                cs.append(0)
            else:
                if t != 0:
                    cs.append(corpus[idx+t])

        contexts.append(cs)

    return np.array(contexts), np.array(targets)

if __name__ == "__main__":
    if sys.argv[1] == "freq":
        text = ""
        for file in tqdm(os.listdir("./data"), desc="Reading Data"):
            with open("./data/"+file, 'r') as f:
                data = f.read()
                data = unicodeToAscii(data)
                tmp = clean_str(data).strip()
                text += tmp + " "

        # corpus = freq(text, 5)
        # print("number of total words: ", len(corpus))
        # print("number of total vocab: ", len(set(corpus)))
        # pickle.dump(corpus, open("./preprocessed_data/freq_corpus", 'wb'), protocol=-1)

        counts = count_freq(text)
        pickle.dump(counts, open("./preprocessed_data/word_cnts", "wb"), protocol=-1)

    elif sys.argv[1] == "preprocess":
        with open('./preprocessed_data/freq_corpus', 'rb') as f:
            text = pickle.load(f)

        corpus, w2id, id2w, counter = preprocess(text)
        preprocessed = [corpus, w2id, id2w, counter]
        print("vocab length : ", len(w2id))
        pickle.dump(preprocessed, open("./preprocessed_data/preprocessed_corpus", 'wb'), protocol=-1)

    elif sys.argv[1] == "subsample":
        with open('./preprocessed_data/preprocessed_corpus', 'rb') as f:
            corpus, w2id, id2w, counter = pickle.load(f)

        window_size = 5
        contexts, targets = subsample_create_contexts_target(corpus, window_size, counter)
        data = [contexts, targets, corpus, w2id, id2w]
        print("numer of training data : ", len(contexts))
        pickle.dump(data, open("./preprocessed_data/data_subsampled", 'wb'), protocol=-1)
        print("Dataset_Created!")

    elif sys.argv[1] == "non-subsample":
        with open('./preprocessed_data/preprocessed_corpus', 'rb') as f:
            corpus, w2id, id2w, counter = pickle.load(f)

        window_size = 5
        contexts, targets = create_contexts_target(corpus, window_size)
        data = [contexts, targets, w2id, id2w]
        print("numer of training data : ", len(contexts))
        pickle.dump(data, open("./preprocessed_data/data_no_subsampled", 'wb'), protocol=-1)
        print("Dataset_Created!")