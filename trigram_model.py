import sys
from collections import defaultdict
import math
import random
import os
import os.path

"""
COMS W4705 - Natural Language Processing - Fall 2022 
Prorgramming Homework 1 - Trigram Language Models
Gurnoor Virdi
"""

def corpus_reader(corpusfile, lexicon=None):
    with open(corpusfile, 'r') as corpus:
        for line in corpus:
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon:
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else:
                    yield sequence


def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence:
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)


def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of n >= 1 
    """

    ngrams = []
    if (n > 1):
        sequence_padding = ["START"] * (n - 1) + sequence + ["STOP"]
        for i in range(len(sequence_padding) - n + 1):
            ngram = tuple(sequence_padding[i:i + n])
            ngrams.append(ngram)

    # this will run if n ==1 because then you do not need to multuply start by 0
    elif (n == 1):
        # print("ran")
        sequence_padding = ["START"] + sequence + ["STOP"]
        for z in range(len(sequence_padding) - n + 1):
            ngram = tuple(sequence_padding[z:z + n])
            ngrams.append(ngram)
    return ngrams


class TrigramModel(object):

    def __init__(self, corpusfile):

        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")

        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)

        #must iterate throguh courpus to get total word count:
        generator = corpus_reader(corpusfile, self.lexicon)
        self.total_word_count = 0
        for sentence in generator:
            sequence_count = len(sentence)
            self.total_word_count += (sequence_count + 1)
        #print(self.total_word_count)

    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """

        self.unigramcounts = defaultdict(int)  # might want to use defaultdict or Counter instead
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)

        ##Your code here

        # go through the lines in the corpus

        for seq in corpus:

            # unigram
            total_unigram = get_ngrams(seq, 1)
            for unigram in total_unigram:
                self.unigramcounts[unigram] += 1

            # bigram
            total_bigram = get_ngrams(seq, 2)
            for bigram in total_bigram:
                self.bigramcounts[bigram] += 1

            # trigram
            total_trigram = get_ngrams(seq, 3)
            for trigram in total_trigram:
                self.trigramcounts[trigram] += 1

        return

    def raw_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        formula is on lecture slide 28
        """
        # the way you calculate this, divide by the number of times the first two instances are used
        if trigram[0:2] == ("START", "START"):
            denom = self.unigramcounts[("START",)]
        else:
            denom = self.bigramcounts[trigram[0:2]]
        if denom != 0:
            unsmoothed_prob = self.trigramcounts[trigram] / denom
            return unsmoothed_prob
        else:
            return 0.0

    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        formula is on lecture slide 28
        """

        unigram = bigram[0:1]
        unsmoothed_prob = 0
        if(self.unigramcounts[unigram] > 0):
            unsmoothed_prob = self.bigramcounts[bigram] / self.unigramcounts[unigram]
        if (unsmoothed_prob != 0):
            return unsmoothed_prob
        else:
            return 0.0

    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        for this divide by the total number of words
        """

        # hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.

        # first calculate all the values in unigramCounts
        # only do this is Trigram Model does not have the instnace of storing the words
        if self.total_word_count > 0:
            unsmoothed_prob = self.unigramcounts[unigram] / self.total_word_count

        if unsmoothed_prob != 0:
            return unsmoothed_prob
        else:
            return 0.0

    def generate_sentence(self, t=20):
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        # blank
        return result

    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1 / 3.0
        lambda2 = 1 / 3.0
        lambda3 = 1 / 3.0

        prob1 = lambda1 * self.raw_trigram_probability(trigram)
        prob2 = lambda2 * self.raw_bigram_probability(trigram[1:3])
        prob3 = lambda3 * self.raw_unigram_probability(trigram[2:3])
        smoothed_prob = prob1 + prob2 + prob3
        #print(smoothed_prob)

        return smoothed_prob

    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """

        # get the trigrams using the getNgrams
        trigrams = get_ngrams(sentence, 3)
        # get the probabilities for the trigrams
        trigram_probability = 0
        for tri in trigrams:
            #print(self.smoothed_trigram_probability(tri))
            #print(tri)
            if(self.smoothed_trigram_probability(tri) >0):
                trigram_probability += math.log2(self.smoothed_trigram_probability(tri))

        return trigram_probability

    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        log_probability = 0.0
        M_tot_words = 0
        for sentence in corpus:
            # add to log_probability
            log_probability += self.sentence_logprob(sentence)
            M_tot_words += len(sentence)

        log_probability_avg = log_probability / M_tot_words
        perplexity = 2**(-log_probability_avg)

        return perplexity


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):
    model1 = TrigramModel(training_file1)
    model2 = TrigramModel(training_file2)

    total = 0
    correct = 0

    for f in os.listdir(testdir1):
        pp_model_1 = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
        pp_model_2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
        total += 1
        if(pp_model_1 < pp_model_2):
            correct += 1

    for f in os.listdir(testdir2):
        pp_model_2 = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
        pp_model_1 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
        total += 1
        if(pp_model_2 < pp_model_1):
            correct += 1

    accuracy = correct/total
    return accuracy


if __name__ == "__main__":
    model = TrigramModel(sys.argv[1])

    ngram_string = ["natural", "language", "processing"]

    output = get_ngrams(ngram_string, 2)
    #print(output)

    count_trigram = model.trigramcounts[('START', 'START', 'the')]
    #print(count_trigram)
    count_bigram = model.bigramcounts[('START', 'the')]
    #print(count_bigram)
    count_unigram = model.unigramcounts[('the',)]
    #print(count_unigram)

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    # Testing perplexity:
    dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    pp = model.perplexity(dev_corpus)
    print(pp)

    # Essay scoring experiment:
    acc = essay_scoring_experiment('train_high.txt', "train_low.txt", "test_high", "test_low")
    print(acc)
