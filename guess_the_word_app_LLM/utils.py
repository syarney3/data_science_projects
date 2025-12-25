import random
from nltk.corpus import wordnet
from random_word import RandomWords 

# Download WordNet if you haven't already
import nltk
nltk.download('wordnet')


def get_word_wordnet():

    # Filter synsets to include only nouns
    noun_synsets = list(wordnet.all_synsets(pos=wordnet.VERB))
    noun_synsets = [x for x in noun_synsets if '_' not in x.name()]
    noun_synsets = [x for x in noun_synsets if '-' not in x.name()]

    # Randomly select one noun synset
    random_noun_synset = random.choice(noun_synsets)

    # Extract the first lemma (one word) for the selected synset
    random_noun = random_noun_synset.lemmas()[0].name()
    random_noun = random_noun.lower()
    #print(f"Random word: {random_noun}")

    return random_noun

# Generate a random word
def generate_word():

    r = RandomWords()
    word = r.get_random_word()
    return word

def find_synonyms(word, hints):

    # Get the synsets for the word
    synsets = wordnet.synsets(word)

    # Extract all synonyms from the synsets
    synonyms = set()  # Using a set to avoid duplicates

    for synset in synsets:
        for lemma in synset.lemmas():
            synonyms.add(lemma.name())

    synonyms = list(synonyms)
    
    # remove already given synonyms
    synonyms = [item for item in synonyms if item not in hints]
    synonyms = [item for item in synonyms if item != word]
    if (len(synonyms) == 0):
        synonym = "No more hints"
    else:
        synonym = synonyms[0].replace("_", " ")

    print(f"The synonym for the word: {word} is: {synonym}")
    print(f"Hints list: {hints}")
    return synonym


