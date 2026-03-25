import re
import random
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud

# tokenizers
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# DATA PREP & WORD CLOUD

docs = []
with open('corpus.txt', 'r', encoding='utf-8') as fl:
    for l in fl:
        l = l.strip()
        if l:  # ignore blank lines
            docs.append(l)

allTokens = []
cleanDocs = []
for doc in docs:
    txt = doc.lower() # lowercase
    txt = re.sub(r'[^a-z\s]', ' ', txt) # keep only letters and spaces
    txt = re.sub(r'\s+', ' ', txt).strip() # remove multiple spaces
    tokens = word_tokenize(txt)
    if tokens:
        cleanDocs.append(tokens)
        allTokens.extend(tokens)

print("Cleaned the corpus\n")
with open('corpus.txt', 'w', encoding='utf-8') as outfile:
    for tokens in cleanDocs: 
        clean = ' '.join(tokens)
        outfile.write(clean + '\n')

vocab = set(allTokens)
vocabSz = len(vocab)
print("Dataset Stats:")
print(f"No of Documents: {len(cleanDocs)}")
print(f"No of Tokens: {len(allTokens)}")
print(f"Vocab Size: {vocabSz}")

txtWordCloud = ' '.join(allTokens)
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis', max_words=100).generate(txtWordCloud)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Most Frequent Words in IIT Jodhpur Corpus", fontsize=16)
plt.tight_layout(pad=0)
cloud_filename = "wordcloud.png" # saving wordcloud as png
plt.savefig(cloud_filename, dpi=300) 
plt.close()
print(f"Word Cloud saved as {cloud_filename}")

# WORD2VEC

sentences = []
vocabCnts = {}
with open('corpus.txt', 'r', encoding='utf-8') as fl:
    for l in fl:
        tokens = l.strip().split()
        if tokens:
            sentences.append(tokens)
            for token in tokens: # counting occurences
                vocabCnts[token] = vocabCnts.get(token, 0) + 1
vocab = list(vocabCnts.keys()) # word to ID mapping
vocabSz = len(vocab)
wordToId = {w: i for i, w in enumerate(vocab)}

print(f"No of sentences: {len(sentences)}. Vocab size: {vocabSz}")

# negative sampling distribution
# using unigram counts raised to the 0.75 power
pow = 0.75
totPow = sum([count ** pow for count in vocabCnts.values()])
probUni = {w: (count ** pow) / totPow for w, count in vocabCnts.items()}

tableSz = 100000
negTable = []
p = 0
for w, prob in probUni.items():
    p += prob
    while len(negTable) < tableSz * p:
        negTable.append(wordToId[w])

# Grid parameters
embeddingDimensions = [50]      
windowSizes = [2, 5]            
negativeSamples = [5, 10]       
learningRate = 0.05
epochs = 10

# to get 300 dimension word embedding for 'research'
# embeddingDimensions = [300]      
# windowSizes = [5]            
# negativeSamples = [10]       
# learningRate = 0.05
# epochs = 10

reportData = []

print("\nGrid search:")

for dim in embeddingDimensions:
    for window in windowSizes:
        for neg in negativeSamples:
            # CBOW
            print(f"Training CBOW (Dim={dim}, Window={window}, Neg={neg})")
            np.random.seed(42)
            # W1 = the actual embeddings we care about, W2 = context weights
            W1Cbow = np.random.uniform(-0.5/dim, 0.5/dim, (vocabSz, dim))
            W2Cbow = np.zeros((vocabSz, dim))
            totLossCbow = 0
            for epoch in range(epochs):
                for sen in sentences:
                    senId = [wordToId[w] for w in sen]
                    for i, trgtId in enumerate(senId):
                        # grab the surrounding words
                        start = max(0, i - window)
                        end = min(len(senId), i + window + 1)
                        cntxtIds = [senId[j] for j in range(start, end) if i != j]
                        if not cntxtIds: continue
                        
                        h = np.mean(W1Cbow[cntxtIds], axis=0) # cbow forward pass: average the context vectors
                        samples = [trgtId]
                        
                        while len(samples) < neg + 1: # pick our negative samples
                            randWrd = random.choice(negTable)
                            if randWrd != trgtId:
                                samples.append(randWrd)
                        
                        # 1 for the real word, 0s for the fake ones
                        labels = np.zeros(len(samples))
                        labels[0] = 1.0
                        scores = np.dot(W2Cbow[samples], h)
                        scores = np.clip(scores, -10, 10) 
                        probs = 1 / (1 + np.exp(-scores))
                        # log loss
                        totLossCbow -= np.log(probs[0] + 1e-9) + np.sum(np.log(1 - probs[1:] + 1e-9))
                        errors = probs - labels
                        # calculate gradients
                        dW2 = np.outer(errors, h)
                        dh = np.dot(errors, W2Cbow[samples])
                        # update weights
                        W2Cbow[samples] -= learningRate * dW2
                        for cId in cntxtIds:
                            W1Cbow[cId] -= learningRate * (dh / len(cntxtIds))
            reportData.append({"Architecture": "CBOW", "Dimensions": dim, "Window Size": window, "Negative Samples": neg, "Loss": round(totLossCbow, 2)})

            # SKIP-GRAM
            print(f"Training Skip-gram (Dim={dim}, Window={window}, Neg={neg})")
            np.random.seed(42)
            W1Sg = np.random.uniform(-0.5/dim, 0.5/dim, (vocabSz, dim))
            W2Sg = np.zeros((vocabSz, dim))
            totLossSg = 0
            for epoch in range(epochs):
                for sen in sentences:
                    senId = [wordToId[w] for w in sen]
                    for i, trgtId in enumerate(senId):
                        # skip-gram uses the center word to predict context
                        h = W1Sg[trgtId]
                        start = max(0, i - window)
                        end = min(len(senId), i + window + 1)
                        cntxtIds = [senId[j] for j in range(start, end) if i != j]
                        for cntxtId in cntxtIds:
                            samples = [cntxtId]
                            while len(samples) < neg + 1:
                                randWrd = random.choice(negTable)
                                if randWrd != trgtId and randWrd != cntxtId:
                                    samples.append(randWrd)
                            labels = np.zeros(len(samples))
                            labels[0] = 1.0
                            scores = np.dot(W2Sg[samples], h)
                            scores = np.clip(scores, -10, 10)
                            probs = 1 / (1 + np.exp(-scores))
                            totLossSg -= np.log(probs[0] + 1e-9) + np.sum(np.log(1 - probs[1:] + 1e-9))
                            errors = probs - labels
                            dW2 = np.outer(errors, h)
                            dh = np.dot(errors, W2Sg[samples])
                            W2Sg[samples] -= learningRate * dW2
                            W1Sg[trgtId] -= learningRate * dh
            reportData.append({"Architecture": "Skip-gram", "Dimensions": dim, "Window Size": window, "Negative Samples": neg, "Loss": round(totLossSg, 2)})

dfRprt = pd.DataFrame(reportData)
print("\nResults Table:")
print("-" * 75)
print(dfRprt.to_string(index=False))
print("-" * 75)


# SEMANTIC ANALYSIS

embeddings = W1Sg  # using final Skip-gram embeddings
norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
nrmlEmbed = embeddings / (norms + 1e-9) # +1e-9 avoids dividing by zero

print("\nTOP 5 NEAREST NEIGHBORS")
qryWrds = ["research", "student", "phd", "examination"]

for word in qryWrds:
    if word not in wordToId:
        print(f"'{word}' isn't in vocab. Skipped.")
        continue
    wordId = wordToId[word]
    wordVec = nrmlEmbed[wordId]
    similarities = np.dot(nrmlEmbed, wordVec) # similarity score against whole vocab
    topInd = np.argsort(similarities)[::-1][:6]
    print(f"\nNeighbors for '{word}':")
    count = 0
    for idx in topInd:
        neighWrd = vocab[idx]
        if neighWrd != word and count < 5:
            print(f"  {count+1}. {neighWrd} (Score: {similarities[idx]:.4f})")
            count += 1

print("\nANALOGY EXPERIMENTS")
analogies = [
    ("ug", "btech", "pg"),
    ("faculty", "teaching", "student"),
    ("bachelor", "btech", "master"),
    ("undergraduate", "ug", "postgraduate"),
    ("btech", "four", "mtech"),
    ("director", "institute", "hod"),
    ("researcher", "research", "teacher"),
    ("phd", "thesis", "btech"),
]

for a, b, c in analogies:
    if a not in wordToId or b not in wordToId or c not in wordToId:
        print(f"Skipping {a}:{b} :: {c}:? - missing words.")
        continue
    vecA = nrmlEmbed[wordToId[a]]
    vecB = nrmlEmbed[wordToId[b]]
    vecC = nrmlEmbed[wordToId[c]]
    # formula: D = B - A + C
    target_vec = vecB - vecA + vecC
    target_vec = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    similarities = np.dot(nrmlEmbed, target_vec)
    topInd = np.argsort(similarities)[::-1][:5]
    print(f"\nAnalogy: {a} : {b} :: {c} : ?")
    for idx in topInd:
        rsltWord = vocab[idx]
        if rsltWord not in [a, b, c]: # make sure not print input words
            print(f"  -> Predicted: {rsltWord} (Score: {similarities[idx]:.4f})")
            break


# PCA VISUALIZATION

wordsPlt = [
    "btech", "mtech", "phd", "ug", "pg", # degress
    "student", "faculty", "director", "hod", "researcher", # roles
    "exam", "theory", "practical", "research", "thesis" # academics
]

validWrds = [w for w in wordsPlt if w in wordToId]
validIds = [wordToId[w] for w in validWrds]

# choosing just the vectors for the words we want to plot
cbowVecs = W1Cbow[validIds]
sgVecs = W1Sg[validIds]

# 50D vectors -> 2D coordinates for plotting
pca = PCA(n_components=2, random_state=42)
cbow2D = pca.fit_transform(cbowVecs)
sg2D = pca.fit_transform(sgVecs)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# CBOW
ax1.scatter(cbow2D[:, 0], cbow2D[:, 1], color='blue', alpha=0.7, edgecolors='k', s=100)
for i, word in enumerate(validWrds):
    ax1.annotate(word, xy=(cbow2D[i, 0], cbow2D[i, 1]), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom', fontsize=10)
ax1.set_title("CBOW Word Embeddings (PCA)", fontsize=14)
ax1.grid(True, linestyle='--', alpha=0.5)

# Skip-gram
ax2.scatter(sg2D[:, 0], sg2D[:, 1], color='red', alpha=0.7, edgecolors='k', s=100)
for i, word in enumerate(validWrds):
    ax2.annotate(word, xy=(sg2D[i, 0], sg2D[i, 1]), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom', fontsize=10)
ax2.set_title("Skip-gram Word Embeddings (PCA)", fontsize=14)
ax2.grid(True, linestyle='--', alpha=0.5)
plt.suptitle("CBOW vs Skip-gram", fontsize=16, y=0.95)
output_img = "pcaClusters.png"
plt.savefig(output_img, dpi=300, bbox_inches='tight')
plt.close()
print(f"Visualization saved as '{output_img}'")

# code to answer google form questions

trgt = "research"
wordId = wordToId[trgt]
vector = W1Sg[wordId]
formVec = ", ".join([f"{val:.4f}" for val in vector])
print(f"{trgt} - {formVec}")
    
top10 = sorted(vocabCnts.items(), key=lambda item: item[1], reverse=True)[:10]
formOp = ", ".join([f"{word}, {count}" for word, count in top10])
print(formOp)