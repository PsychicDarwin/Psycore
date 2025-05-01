# from bert_score import BERTScorer
# from transformers import BertTokenizer, BertForMaskedLM, BertModel
# from evaluate import load
#  from rouge_score import rouge_scorer

# class EvaluationProcess(ProcessFlow):

#     def prep_values(vector_store, user_prompt, response):
#         doc, _ = vector_store.similarity_search_with_score(query=user_prompt, k=10)[0]
#         reference = doc.page_content
#         candidate = response
#         return reference, candidate

#     def calculate_bertscore(reference, candidate):
#         bertscore = load("bertscore")
#         bert_results = bertscore.compute(predictions=[candidate], references=[reference], lang="en")
    
#     def calculate_rouge(reference, candidate):
#         scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
#         rouge_results = scorer.score(reference, candidate)

from evaluate import load
from rouge_score import rouge_scorer

class EvaluationProcess:
    def __init__(self):
        """Initialize the evaluation process by loading required metrics."""
        self.bertscore = load("bertscore")
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    def prep_values(self, vector_store, user_prompt, response):
        """Retrieve reference text based on user prompt."""
        doc, _ = vector_store.similarity_search_with_score(query=user_prompt, k=10)[0]
        return doc.page_content, response

    def calculate_bertscore(self, reference, candidate):
        """Compute BERTScore Precision, Recall, and F1."""
        return self.bertscore.compute(predictions=[candidate], references=[reference], lang="en")

    def calculate_rouge(self, reference, candidate):
        """Compute ROUGE-L score."""
        return self.rouge_scorer.score(reference, candidate)