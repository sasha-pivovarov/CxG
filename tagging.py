import spacy
import pandas as pd
from nltk.corpus import wordnet

verbs = ["agonize", "fret", "obsess", "worry", "grieve"]
conj = ["over", "about"]
proc = spacy.load("en") # aku seorang kapiten
with open("raw2.txt", "r") as io:
    phrases = io.read().split('\n')

df = pd.DataFrame({"text": phrases})
res = [proc(phrase) for phrase in df["text"]]

for result in res[:5]:
    print("====================================")
    print(result.text)
    for token in result:
        if token.text == "over":

            print(list(token.children))
            print(list(token.subtree))
            print("####################")
            print([t.dep_ for t in token.subtree])
            print("####################")
            print(list(token.ancestors))
            for child in token.children:
                print(child.dep_)
            print(token.right_edge)
            print(token.left_edge)
            print("====================================")

df["DEP_VAL"] = None
df["DEP_TAG"] = None
df["DEP_STR"] = None
df["DEP_POS"] = None
df["VERB_STR"] = None
df["VERB_DEP"] = None
df["TARGET"] = None

skipped = 0
for i in range(len(res)):
    for token in res[i]:
        if token.text in conj and token.nbor(-1).lemma_ in verbs:
            target = token.text
            dep_num = len(list(token.children))
            if dep_num != 1:
                print(list(token.children))
                skipped += 1
                continue # phrasal
            dep = list(token.children)[0]
            dep_val = dep.dep_
            verb = None
            for anc in token.ancestors:
                if anc.pos_ == "VERB":
                    verb = anc
                    break

            if verb == None:
                continue
            df["DEP_VAL"][i] = dep_val
            df["DEP_TAG"][i] = dep.tag_
            df["DEP_POS"][i] = dep.pos_
            df["DEP_STR"][i] = dep.lemma_
            df["VERB_STR"][i] = verb.lemma_
            df["VERB_DEP"][i] = verb.dep_
            df["TARGET"][i] = target

print("Skipped %d" % skipped)
df.to_csv("tagged2.csv")

