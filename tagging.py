import spacy
import pandas as pd

proc = spacy.load("en") # aku seorang kapiten
with open("raw.txt", "r") as io:
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
            print(list(token.ancestors))
            for child in token.children:
                print(child.dep_)
            print("====================================")

df["DEP"] = None
df["DEP_TAG"] = None
df["DEP_STR"] = None
df["DEP_POS"] = None
df["VERB_STR"] = None


for i in range(len(res)):
    for token in res[i]:
        if token.text == "over":
            dep_num = len(list(token.children))
            if dep_num != 1:
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
            df["DEP"][i+1] = dep_val
            df["DEP_TAG"][i+1] = dep.tag_
            df["DEP_POS"][i+1] = dep.pos_
            df["DEP_STR"][i+1] = dep.text
            df["VERB_STR"][i+1] = verb.text

df.to_csv("tagged.csv")

