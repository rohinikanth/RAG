from transformers import DPRContextEncoder, DPRContextEncoderTokenizer, DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import torch

def load_context_encoder():
    tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
    model = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
    return tokenizer, model

def load_question_encoder():
    tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
    model = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
    return tokenizer, model

