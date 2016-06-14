from wmd import get_wmd_distance
from gensim import utils
import random

class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                	self.sentences += [line]
        return self.sentences

length = 2000

# 'data/train-neg.txt':'TRAIN_NEG', 
sources = {'data/train-pos.txt':'TRAIN_POS'}
# sentences = LabeledLineSentence(sources).to_array()
sentences = random.sample(LabeledLineSentence(sources).to_array(), 12500)
sentences = [line[:length] for line in sentences]


sample1 = "this show was incredible i ve seen all three and this is the best this movie has suspense a bit of romance stunts that will blow your mind go bobbie great characters and amazing locations where was this filmed will there be more i really liked the story line with her brother looking forward to chameleon and to see how the world is saved yet again"
sample2 = "this anime was underrated and still is hardly the dorky kids movie as noted i still come back to this years after i first saw it one of the better movies released the animation while not perfect is good camera tricks give it a d feel and the story is still as good today even after i grew up and saw ground breakers like neon genesis evangelion and rahxephon it has nowhere near the depth obviously but try to see it from a lighthearted view it s a story to entertain not to question still one of my favourites i come back too when i feel like a giggle on over more lighthearted animes not to say its a childish movies there are surprisingly sad moments in this and you need a sense of humour to see it all"

target = sample1[:length]


scores = {}
for index, sentence in enumerate(sentences):
	if index % 100 == 0: print index
	scores[index] = get_wmd_distance(target, sentence)

sorted_list = sorted(scores.items(), key=lambda kv: kv[1])
top = sorted_list[:20]

print target
for index, score in top:
    print "\n"
    print sentences[index]
