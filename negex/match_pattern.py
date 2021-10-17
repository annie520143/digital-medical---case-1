from wrapper import *

def weighting(sentenceList):
    
    score = 0
    for i, sentence in enumerate(sentenceList):
        #no negated, affirmed
        if(sentence.lower().rfind('negated') == -1): 
            #predict 1
            score = score+1
        else: 
            #predict 0
            score = score-1

    if score <= 0:
        return 0
    else: return 1

def negexFormatting(root, txts, keys, keywords, labels):
    filename = root + 'negex.txt' 
    tagDict = ["Negated", "Affirmed"] 
    predicts = []

    with open(filename, 'w') as f:
        f.write("Report No.\t")
        f.write("Concept Sentence\tNegation\n")

        #ith record
        for i, data in enumerate(txts):
            #for ith record, which sentence contain target
            for j, key in enumerate(keys[i]):

                if (labels!=[]): tag = tagDict[labels[i]]
                #Validation Case
                else: tag = "Affirmed"
                f.write(str(i) + "\t" + keywords[i][j] + "\t" + data[key] + "\t" + tag + "\n")
                
    f.close()
    negexWrapper()

    with open("negex_output.txt", 'r') as f:
        output = [line.strip() for line in f if line.strip()]
    f.close()
    cur = 1

    for i in range(len(txts)):
        #no keywords in sentence, directly classify to non-obesity
        if(keys[i] == []): predicts.append(0)
        else:
            end = cur+len(keys[i])
            predicts.append(weighting(output[cur:end]))
            cur = end
    return predicts