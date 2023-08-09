import  utils
import  numpy       as      np
import  pandas      as      pd


def read_dataset(dataset_path:list,):
    df = pd.DataFrame(dataset_path)
    return df

def normalize_sentence(df:pd.core.frame.DataFrame,):
    """    
    The normalize_sentence function takes a dataframe df and a Container class as input and normalizes the sentences 
    in the given language. 
    -------------------------------
        * The function converts the sentences to lowercase 
        * removes any non-alphabetic characters using a regular expression
        * It then normalizes the text to Unicode NFD (Normalization Form Decomposition) 
        and encodes it to ASCII to remove any non-ASCII characters
        * it decodes the text back to UTF-8 format and returns the normalized sentence.
    -------------------------------    
    Overall, this function is useful for standardizing the text and removing any unwanted characters that may 
    interfere with downstream natural language processing tasks such as text classification or sentiment analysis.
    However, it is important to note that this normalization technique may not be suitable for all languages or text
    types, and more advanced normalization techniques may be required in some cases.
    -------------------------------
    this class is inspiered from prepareData & filterpair in 
    https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html & 
    https://www.guru99.com/seq2seq-model.html
    """
    sentence = df[0].str.lower()
    sentence = sentence.str.replace('[^A-Za-z\s]+', '',regex=True)
    sentence = sentence.str.normalize('NFD')
    sentence = sentence.str.encode('ascii', errors='ignore').str.decode('utf-8')
    return sentence


def process_data(df:pd.core.frame.DataFrame,source):
    """
        Main tokenizer funtion 
        ---> in case of future changes in the Container remember to to change the class in here to 
    """
    print("Read %s sentence pairs" % len(df))
    sentence = normalize_sentence(df)
    

    for i in range(len(df)):
            source.addSentence(sentence[i])
    return source