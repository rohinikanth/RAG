from Utils.extract import extract_paragraphs_from_pdf
from Utils.plot import plots
from Utils.encode import encode_contexts
from Models.dpr_encoder import load_context_encoder, load_question_encoder

import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_answer(question, contexts, model, tokenizer):
    context_text = " ".join(contexts)[:1800]
    input_text = question + " " + context_text
    inputs = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=1024, padding='max_length')
    output_ids = model.generate(inputs['input_ids'], max_new_tokens=100, num_beams=4, early_stopping=True)
    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output.split("Answer:")[-1].strip() if "Answer:" in output else output.strip()

def main():
    # Load PDF and extract paragraphs
    pdf_path = 'data/s41598-021-03585-1.pdf'
    paragraphs = extract_paragraphs_from_pdf(pdf_path)
    print(f"Total unique paragraphs: {len(paragraphs)}")

    # Load DPR Encoders
    context_tokenizer, context_encoder = load_context_encoder()
    question_tokenizer, question_encoder = load_question_encoder()

    # Encode all context paragraphs
    context_embeddings = encode_contexts(paragraphs, context_tokenizer, context_encoder)

    # Optional: visualize 10 embeddings
    plots(context_embeddings[20:30])

    # Build FAISS index
    embedding_dim = context_embeddings.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(context_embeddings.astype('float32'))

    # Encode a question
    question = "What is the architecture used in this model? Explain"
    q_inputs = question_tokenizer(question, return_tensors='pt')
    q_embed = question_encoder(**q_inputs).pooler_output.detach().numpy()

    # Search top 5 similar paragraphs
    _, I = index.search(q_embed.astype('float32'), k=5)
    top_contexts = [paragraphs[i] for i in I[0]]

    # Load GPT-2 for answer generation
    gpt_tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    gpt_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    if gpt_tokenizer.pad_token is None:
        gpt_tokenizer.pad_token = gpt_tokenizer.eos_token
    gpt_model.resize_token_embeddings(len(gpt_tokenizer))
    gpt_model.generation_config.pad_token_id = gpt_tokenizer.pad_token_id

    # Generate and print answer
    answer = generate_answer(question, top_contexts, gpt_model, gpt_tokenizer)
    print("Generated Answer:\n", answer)

if __name__ == "__main__":
    main()
