from    nltk.translate.bleu_score   import  sentence_bleu,corpus_bleu, SmoothingFunction
#corpus_bleu
def calculate_bleu_score(reference, hypothesis):
    smoothie = SmoothingFunction()  # Smoothing method for BLEU score calculation
    # Convert reference and hypothesis sequences to lists of strings
    reference = [[str(token) for token in ref] for ref in reference]
    hypothesis = [[str(token) for token in hyp] for hyp in hypothesis]
    # Calculate BLEU score
    return sentence_bleu(reference, hypothesis, smoothing_function=smoothie)#,weights = (0.5, 0.5, 0.0, 0.0)