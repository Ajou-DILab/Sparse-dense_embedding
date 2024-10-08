import pandas as pd
from nltk.corpus import wordnet as wn

# word_class element: NOUN, VERB, ADJ, ADV
word_class = 'word_class element name'

wc_name = getattr(wn,word_class)

words_set = set()

for synset in wn.all_synsets(wc_name):
    for lemma in synset.lemmas():
        words_set.add(lemma.name())

wn_df = pd.DataFrame(columns=['word','definition','example_sentence','masking_sentence'])

def word_masking(w,sen):
  mask_sen = sen.replace("'"," ")
  w = w.replace('_',' ')
  sen_token_list = mask_sen.split(' ')
  for tok in sen_token_list:
    #print(tok)
    #print(wn.morphy(tok))
    if wn.morphy(tok) == None:
        if w == tok:
          mask_sen = mask_sen.replace(tok,'[MASK]')
    else:
        if w.lower() == wn.morphy(tok):
        #print(w.lower())
        #print(wn.morphy(tok))
          mask_sen = mask_sen.replace(tok,'[MASK]')
        else:
          continue
  return mask_sen

def get_meanings_and_examples(wset, df):
  words = []
  meanings = []
  sens = []
  mask_sens = []
  for word in wset:
    synsets = wn.synsets(word)
    if not synsets:
        print(f"Word '{word}' not found in WordNet.")
        return

    for synset in synsets:
        #print(synset)
        lemma_words = []
        for lemma in synset.lemmas():
            lemma_words.append(lemma.name())

        # print(lemma_words)
        #print(f"Meaning: {synset.definition()}")
        
        examples = synset.examples()
        if examples:
            for example in examples:
                for w in lemma_words:
                    if w in example:
                        words.append(w)
                        meanings.append(synset.definition())
                        sens.append(example)
                        mask_sens.append(word_masking(w,example))
                    else:
                        continue
  df['word'] = words
  df['definition'] = meanings
  df['example_sentence'] = sens
  df['masking_sentence'] = mask_sens

  return df


