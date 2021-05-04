from nltk import tokenize

def split_into_sentences(text):
    return tokenize.sent_tokenize(text)

def split_into_words(text):
    return tokenize.word_tokenize(text)

def translate_text(text):
    sentences = split_into_sentences(text)

    output = []
    for s in sentences:

        words = split_into_words(s)

        t_s = []
        for w in words:
            t_w = w
            t_s.append(t_w)

        output.append(' '.join(t_s))

    return output