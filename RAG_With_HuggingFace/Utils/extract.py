import re
import spacy
from PyPDF2 import PdfReader

nlp = spacy.load("en_core_web_sm")

def extract_paragraphs_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + " "

    full_text = re.sub(r'-\s+', '', full_text)
    full_text = re.sub(r'\s+', ' ', full_text)

    doc = nlp(full_text)
    sentences = [sent.text.strip() for sent in doc.sents]

    paragraphs = []
    paragraph = ""
    for sentence in sentences:
        if len(paragraph) + len(sentence) < 500:
            paragraph += " " + sentence
        else:
            if paragraph.strip():
                paragraphs.append(paragraph.strip())
            paragraph = sentence
    if paragraph.strip():
        paragraphs.append(paragraph.strip())

    paragraphs = [p for p in paragraphs if len(p) > 70]
    return list(set(paragraphs)) 
