import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.

from haystack import Finder
from haystack.document_store.faiss import FAISSDocumentStore
from haystack.preprocessor.cleaning import clean_wiki_text
from haystack.preprocessor.utils import convert_files_to_dicts, fetch_archive_from_http
from haystack.reader.farm import FARMReader
from haystack.utils import print_answers
from haystack.retriever.dense import DensePassageRetriever

# ## Preprocessing of documents
# Let's first get some documents that we want to query
@st.cache(
    
)
def create_store():
	# FAISS is a library for efficient similarity search on a cluster of dense vectors.
	# The FAISSDocumentStore uses a SQL(SQLite in-memory be default) document store under-the-hood
	# to store the document text and other meta data. The vector embeddings of the text are
	# indexed on a FAISS Index that later is queried for searching answers.
	document_store = FAISSDocumentStore()

# FAISS is a library for efficient similarity search on a cluster of dense vectors.
# The FAISSDocumentStore uses a SQL(SQLite in-memory be default) document store under-the-hood
# to store the document text and other meta data. The vector embeddings of the text are
# indexed on a FAISS Index that later is queried for searching answers.
document_store = FAISSDocumentStore()

# ## Preprocessing of documents
# Let's first get some documents that we want to query
doc_dir = "data/article_txt_got"
s3_url = "https://s3.eu-central-1.amazonaws.com/deepset.ai-farm-qa/datasets/documents/wiki_gameofthrones_txt.zip"
fetch_archive_from_http(url=s3_url, output_dir=doc_dir)

# convert files to dicts containing documents that can be indexed to our datastore
dicts = convert_files_to_dicts(dir_path=doc_dir, clean_func=clean_wiki_text, split_paragraphs=True)

# Now, let's write the docs to our DB.
document_store.write_documents(dicts)

### Retriever
retriever = DensePassageRetriever(document_store=document_store,
                                  query_embedding_model="facebook/dpr-question_encoder-single-nq-base",
                                  passage_embedding_model="facebook/dpr-ctx_encoder-single-nq-base",
                                  max_seq_len_query=64,
                                  max_seq_len_passage=128,
                                  batch_size=32,
                                  use_gpu=True,
                                  embed_title=True,
                                  use_fast_tokenizers=True
                                  )

# Important:
# Now that after we have the DPR initialized, we need to call update_embeddings() to iterate over all
# previously indexed documents and update their embedding representation.
# While this can be a time consuming operation (depending on corpus size), it only needs to be done once.
# At query time, we only need to embed the query and compare it the existing doc embeddings which is very fast.
document_store.update_embeddings(retriever)

### Reader
# Load a  local model or any of the QA models on
# Hugging Face's model hub (https://huggingface.co/models)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)

### Finder
# The Finder sticks together reader and retriever in a pipeline to answer our actual questions.
finder = Finder(reader, retriever)


#prediction = finder.get_answers(question="Who is the father of Arya Stark?", top_k_retriever=10, top_k_reader=5)


# prediction = finder.get_answers(question="Who created the Dothraki vocabulary?", top_k_reader=5)
# prediction = finder.get_answers(question="Who is the sister of Sansa?", top_k_reader=5)
#finder = create_retriver()
st.title('Narad: End-to-End Question Answering Pipeline')

# Start sidebar
header_html = "Options"
header_full = """
<html>
  <head>
	<style>
	  .img-container {
		padding-left: 90px;
		padding-right: 90px;
		padding-top: 50px;
		padding-bottom: 50px;
		background-color: #f0f3f9;
	  }
	</style>
  </head>
  <body>
	<span class="img-container"> <!-- Inline parent element -->
	  %s
	</span>
  </body>
</html>
""" % (
	header_html,
)
st.sidebar.markdown(
	header_full,
	unsafe_allow_html=True,
)

# Long Form QA with ELI5 and Wikipedia
description = """
This demo presents End-to-End system for question answering based on articles on Games of Thrones!
"""
st.sidebar.markdown(description, unsafe_allow_html=True)

action_list = [
	"Answer the question",
	"View the retrieved document only",
	"Show me everything, please!",
]
demo_options = st.sidebar.checkbox("Demo options")
if demo_options:
	action_st = st.sidebar.selectbox(
		"",
		action_list,
		index=2,
	)
	action = action_list.index(action_st)
	show_type = st.sidebar.selectbox(
		"",
		["Show probability ", "Show score"],
		index=0,
	)
	
	
else:
	action = 2
	show_passages = True

choose_retrvr = st.sidebar.checkbox("Retriever options")
if choose_retrvr:
	retriver = st.sidebar.selectbox(
			"",
			["BM25","Dense","Tf-Idf","Embedding"],
			index=1,
		)

mod_op = st.sidebar.checkbox("Model options")
if mod_op:
	mod = st.sidebar.selectbox(
			"",
			["Abstractive", "Genrative type(RAG)"],
			index=0,
		)
#prediction = finder.get_answers(question="Who is the father of Arya Stark?", top_k_retriever=10, top_k_reader=5)
st.checkbox("Want to upload a articel/document")
file = st.file_uploader("Choose a file(.pdf, .dox, .csv, .txt)")
# prediction = finder.get_answers(question="Who created the Dothraki vocabulary?", top_k_reader=5)
# prediction = finder.get_answers(question="Who is the sister of Sansa?", top_k_reader=5)
# start main text
questions_list = [
	"<MY QUESTION>",
	"Who is the father of Arya Stark?",
	"Who created the Dothraki vocabulary?",
	"Who is the sister of Sansa?",

]
question_s = st.selectbox(
	"What would you like to ask? ---- select <MY QUESTION> to enter a new query",
	questions_list,
	index=1,
)
if question_s == "<MY QUESTION>":
	question = st.text_input("Enter your question here:", "")
else:
	question = question_s

if st.button("Show me!"):
	if action in [0, 1, 2]:
		prediction = finder.get_answers(question=str(question), top_k_retriever=10, top_k_reader=5)
		answers = [prediction['answers'][i]['answer'] for i in range(len(prediction['answers']))]
		contexts = [prediction['answers'][i]['context'] for i in range(len(prediction['answers']))]
		probabilities = [prediction['answers'][i]['probability'] for i in range(len(prediction['answers']))]

		#answers =["sg","SDgsdg"]
		#contexts = ["sg","SDgsdg"]
	if action in [0, 2]:
		
		st.markdown("### The model generated answer is:")
		st.write(answers[0])
		st.markdown("--- \n ### The model extaracted answer from following passage:")
		st.write(contexts[0])
		st.markdown("--- \n #### Confidence:")
		st.write(probabilities[0])
	if action == 2 :
		st.markdown("--- \n ### The other answers and corresponding passages:")
		for ans, con, prob in zip(answers[1:],contexts[1:],probabilities[1:]):
			st.write(ans)
			st.write(
					'> <span style="font-family:arial; font-size:10pt;">' + con + "</span>", unsafe_allow_html=True
				)
			st.markdown("--- \n #### Confidence:")
			st.write(probabilities[0])
	if action in [1]:
		st.markdown("--- \n ### The related passages to question are:")
		for ans, con in zip(answers,contexts):
			st.write(
					'> <span style="font-family:arial; font-size:10pt;">' + con + "</span>", unsafe_allow_html=True
				)

