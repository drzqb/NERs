import json

with open("data/OriginalFiles/train_span.txt", "r", encoding="utf-8") as f:
    all_datas = json.load(f)

alllen = {}
for data in all_datas:
    text = data["context"]
    seqlen = len(text)

    if seqlen not in alllen:
        alllen[seqlen] = 1
    else:
        alllen[seqlen] += 1

for k in sorted(alllen.keys()):
    print(k, ": ", alllen[k])
