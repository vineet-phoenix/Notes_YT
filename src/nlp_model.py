import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class VideoAIAssistant:
    def __init__(self):
        # Automatically use GPU if available
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "google/flan-t5-base"
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)

    def chunk_text(self, text, chunk_size=1200):
        """Splits long transcripts into manageable chunks to prevent memory crashes."""
        sentences = text.split(". ")
        chunks, current_chunk = [], ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "

        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks

    def summarize_video(self, transcript_text):
        """Generates summary notes from the full transcript."""
        chunks = self.chunk_text(transcript_text)
        notes = []

        for chunk in chunks:
            prompt = f"Summarize the following text clearly:\n{chunk}"
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
            
            summary_ids = self.model.generate(
                **inputs, 
                max_new_tokens=120, 
                num_beams=4, 
                length_penalty=1.0, 
                early_stopping=True
            )
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            notes.append(summary)
            
        return "\n- ".join(notes)

    def answer_question(self, context, question):
        """Uses the generated notes as context to answer user questions."""
        prompt = f"Read the context and answer the question.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(self.device)
        output_ids = self.model.generate(**inputs, max_new_tokens=100)
        
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
