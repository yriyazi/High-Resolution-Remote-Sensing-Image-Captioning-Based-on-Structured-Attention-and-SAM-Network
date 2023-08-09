# import operator


class Countainer:
    """
        The Countainer class is designed to create a dictionary of unique words with their corresponding indices for 
    natural language processing tasks. It automates the counting of words in a text and adds the SOS and EOS 
    tokens to each sentence for tokenization purposes.
    
    -------------------------------
        this class is inspiered from Lung in https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html & 
    https://www.guru99.com/seq2seq-model.html & 
    https://medium.com/@tarunjethwani31/using-glove-word-embeddings-with-seq2seq-encoder-decoder-in-pytorch-560a3940242
    """    
    def __init__(self,):
        """
        Attributes:
            word2idx (dict): A dictionary that maps each word to its corresponding index.
            word2count (dict): A dictionary that counts the frequency of each word in the text.
            index2word (dict): A dictionary that maps each index to its corresponding word.
            n_words (int): A counter that keeps track of the total number of unique words in the text. 

        """
        self.word2index   = {"<pad>": 0   , "sos": 1, "eos": 2}
        # self.word2count = {}
        self.index2word = {0 : "<pad>"  ,1: "sos" ,  2: "eos"}
        self.n_words = 3                                     
        
    def addSentence(self, sentence):
        """
        Splits the sentence into individual words and adds each word to the container using the addWord() method.

        Args:
            sentence (str): The input sentence.

        """
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        """
        Adds a word to the container and updates its count.

        Args:
            word (str): The word to add.

        """
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            # self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        # else:
            # self.word2count[word] += 1
