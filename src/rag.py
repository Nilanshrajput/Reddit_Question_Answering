from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration 

tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq") 
retriever = RagRetriever.from_pretrained("facebook/rag-sequence-nq", dataset="wiki_dpr", index_name='compressed') 
model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq", retriever=retriever) 

input_dict = tokenizer.prepare_seq2seq_batch("how many countries are in europe", return_tensors="pt") 

generated = model.generate(input_ids=input_dict["input_ids"]) 
print(tokenizer.batch_decode(generated, skip_special_tokens=True)[0]) 
