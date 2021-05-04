import torch
from torch.autograd import Variable

from nltk import tokenize

from constants import device

from utils import model, english_lang, tamil_lang

def split_into_sentences(text):
    return tokenize.sent_tokenize(text)

def split_into_words(text):
    return tokenize.word_tokenize(text)

def translate_text(text):
    sentences = split_into_sentences(text)

    output = []
    for s in sentences:
        t_s = eng_to_tam_converter(sentence)

        output.append(' '.join(t_s))

    return output

def eng_to_tam_converter(sentence):
    #split the input sentence
    sentence = sentence.split()
    translated_sentence=""
    #converting the word to index
    indexed = []
    for tok in sentence:
        try:
            if english_lang.word2index[tok] != 0 :
                indexed.append(english_lang.word2index[tok])
            else:
                indexed.append(0)
        except:
            print(tok,"is not in vocab")
    
    #convert the indexes into tensor
    sentence = Variable(torch.LongTensor([indexed])).to(device)
    
    #first token to the decoder
    trg_init_tok = english_lang.word2index["SOS"]
    trg = torch.LongTensor([[trg_init_tok]]).to(device)
    translated_sentence = ""
    print("Converting...")

    #max embedding size
    maxlen = 150
    for i in range(maxlen):
        #predict the target word
        pred = model(sentence.transpose(0,1), trg)
        #convert the predicted index to word
        add_word = tamil_lang.index2word[pred.argmax(dim=2)[-1].item()] 
        print(add_word)
        #end when we got EOS (end of sentence)
        if add_word=="EOS":
            break
        #update the translated_sentence
        translated_sentence+=" "+add_word
        #pass the predict word to the decoder as input
        trg = torch.cat((trg,torch.LongTensor([[pred.argmax(dim=2)[-1]]]).cuda()))
    print("\nConverted sentence:-")
    return translated_sentence