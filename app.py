import streamlit as st
import os
try:
    from PIL import Image
except ImportError:
    import Image

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import csv
import bs4
import requests
import pytesseract
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode
from deep_translator import GoogleTranslator
import spacy as spacy
from spacy import displacy
import spacy_streamlit
import streamlit.components.v1 as components
from spacy_streamlit import visualize_ner			
from spacy.matcher import Matcher 
from spacy.tokens import Span 
import networkx as nx
from pyvis.network import Network
import nltk
nltk.download('punkt')
from nltk import tokenize
nlp = spacy.load('en_core_web_sm')
# Summary Pkgs
from gensim.summarization import summarize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from pyvis.network import Network


PAGE_CONFIG = {"page_title":"MFA.Analyser","page_icon":"https://toppng.com//public/uploads/preview/flag-of-egypt-eagle-115629057682qouw8kymo.png","layout":"centered"}
st.set_page_config(**PAGE_CONFIG)




def main():
	st.title("MFA Document Analyser")
	st.subheader("Just Upload your new Document, then get out analytics on the content alone and with comparison with the closest documents to it.")

  # Read pervious documents
	df= pd.read_csv('./Affairs_output2.csv', index_col= 0)

	with st.sidebar.container():
		st.image("https://www.egyptconsulates.org/wwwroot/images/logo.svg", use_column_width=True)

	menu = ["Home","About"]
	choice = st.sidebar.selectbox('Menu',menu)
	if choice == 'Home':
		st.subheader("Main Page")	

	uploaded_files = st.file_uploader("Upload a file" , type=("png", "jpg","jpeg"), accept_multiple_files=True)
	
	if uploaded_files:

		img_option = st.selectbox('Apply processing on only one of the uploaded files.', tuple(uploaded_files))

		with st.spinner("Applying OCR on the img...."):
				# Upload Document Image
				image = Image.open(img_option)
				st.image(image, caption='Uploaded Image.', use_column_width=True)

				# Apply OCR on the image
				custom_config = r'-l eng+ara --psm 6'
				extractedInformation = pytesseract.image_to_string(image, config=custom_config)
				Arabic_text = ' '.join(re.findall(r'[\u0600-\u06FF0-9_]+',extractedInformation))
				
				# translate to English as Spacy not defined with arabic lang
				translated_doc = GoogleTranslator(source='auto', target='en').translate(Arabic_text)     


		# Append the new document to the main dataframe
		options = ['English Document', 'Arabic Document']
		selection = st.sidebar.selectbox("Choose OCR result's Language", options , index=1)
		if selection == 'English Document':
				st.success(translated_doc, icon="✅")
		elif selection == 'Arabic Document':
				st.success(Arabic_text, icon="✅")

			#############################################################
		# Function for Sumy Summarization
		# src: https://blog.jcharistech.com/2019/11/28/summarizer-and-named-entity-checker-app-with-streamlit-and-spacy/
		def sumy_summarizer(docx):
			parser = PlaintextParser.from_string(docx,Tokenizer("english"))
			lex_summarizer = LexRankSummarizer()
			summary = lex_summarizer(parser.document,3)
			summary_list = [str(sentence) for sentence in summary]
			result = ' '.join(summary_list)
			return result

		# Summaryzer Streamlit App
		st.title("Document Summaryzer")
		st.subheader("Summarize text of inserted Document")
		summarizer_type = st.selectbox("Summarizer Type",["Gensim","Sumy Lex Rank"])
		if st.button("Summarize"):
			if summarizer_type == "Gensim":
				summary_result = summarize(translated_doc)
			elif summarizer_type == "Sumy Lex Rank":
				summary_result = sumy_summarizer(translated_doc)

			st.write(summary_result)
			#############################################################

		new_row = {'Document_translated':translated_doc}
		df2 = df.append(new_row, ignore_index=True)

		result = []
		def sentences_similartity(df,doc1,doc2):
			df['new_doc'] = translated_doc
			for i in df.index :  
				similarity_Ratio = nlp(df[doc1][i]).similarity(nlp(df[doc2][i]))
				result.append(similarity_Ratio)
			df['similarity_Ratio'] = result 
			return(df)

		# Save similarity output in separate container
		with st.container():
			st.title("Similarity Matcher")
			st.subheader("Most Similar Documents")
			Most_sim_num = st.slider('Select the number of most similar documents ...', 0, 10, 5)

			new_df = sentences_similartity(df2,'new_doc','Document_translated')
			new_df = new_df.nlargest(Most_sim_num, ['similarity_Ratio'])[['Document_translated','similarity_Ratio']]
			AgGrid(new_df)

			with st.expander("See explanation"):
				st.write("The table above shows most similar documents to the document you've just uploaded. \n It's supposed to be discussing *one Topic* You can download it and give it a look.")
				st.image("https://cdn0.iconfinder.com/data/icons/flatie-action/24/action_005-information-detail-notification-alert-512.png" , width=100)

		# NER : https://github.com/explosion/spacy-streamlit
		input_text = nlp(translated_doc)
		visualize_ner(input_text, labels=nlp.get_pipe("ner").labels)
		
			##########################################################################################################################

		# Start Networkx
		Doc_text = tokenize.sent_tokenize(translated_doc)
		def get_entities(sent):   
			## chunk 1
				ent1 = ""
				ent2 = ""

				prv_tok_dep = ""    # dependency tag of previous token in the sentence
				prv_tok_text = ""   # previous token in the sentence

				prefix = ""
				modifier = ""
			
				for tok in nlp(sent):
						## chunk 2
						# if token is a punctuation mark then move on to the next token
						if tok.dep_ != "punct":
									# check: token is a compound word or not
								if tok.dep_ == "compound":
										
										prefix = tok.text
						# if the previous word was also a 'compound' then add the current word to it
						if prv_tok_dep == "compound":
									prefix = prv_tok_text + " "+ tok.text
					
					# check: token is a modifier or not
						if tok.dep_.endswith("mod") == True:
								modifier = tok.text
						# if the previous word was also a 'compound' then add the current word to it
						if prv_tok_dep == "compound":
								modifier = prv_tok_text + " "+ tok.text
					
					## chunk 3
						if tok.dep_.find("subj") == True:
								ent1 = modifier +" "+ prefix + " "+ tok.text
								prefix = ""
								modifier = ""
								prv_tok_dep = ""
								prv_tok_text = ""      

					## chunk 4
						if tok.dep_.find("obj") == True:
								ent2 = modifier +" "+ prefix +" "+ tok.text
						
					## chunk 5  
					# update variables
						prv_tok_dep = tok.dep_
						prv_tok_text = tok.text
				return [ent1.strip(), ent2.strip()]

		entity_pairs = []
		for i in Doc_text:
			entity_pairs.append(get_entities(i))
				
		#st.write(entity_pairs)
			#############################################################

		def get_relation(sent):

				doc = nlp(sent)

			# Matcher class object 
				matcher = Matcher(nlp.vocab)

			#define the pattern 
				pattern = [{'DEP':'ROOT'}, 
								{'DEP':'prep','OP':"?"},
								{'DEP':'agent','OP':"?"},  
								{'POS':'ADJ','OP':"?"}] 

				matcher.add("matching_1",[pattern]) 

				matches = matcher(doc)
				k = len(matches) - 1 

				span = doc[matches[k][1]:matches[k][2]] 

				return(span.text)

		relations = []
		for i in Doc_text:
			try:
					relations.append(get_relation(i))
			except:
					relations.append("")

		st.title("Graph Network")
		st.subheader("Graph generations from the input image's text.")		

		#st.write(relations)
		# extract subject
		source = [i[0] for i in entity_pairs]

		# extract object
		target = [i[1] for i in entity_pairs]

		kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relations})
		st.write(kg_df)

		# create a directed-graph from a dataframe
		#G=nx.from_pandas_edgelist(kg_df, "source", "target", edge_attr=True, create_using=nx.MultiDiGraph())
		#fig, ax = plt.subplots()
		#pos = nx.spring_layout(G)
		#nx.draw(G, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos = pos)
		#st.pyplot(fig)
			##########################################################################################################################
		# Source : https://github.com/napoles-uach/streamlit_network

		def simple_func(physics): 
			nx_graph =nx.from_pandas_edgelist(kg_df, "source", "target", edge_attr=True, create_using=nx.MultiDiGraph())
			nt = Network("500px", "899px",notebook=True)
			nt.from_nx(nx_graph)

			if physics:
				nt.show_buttons(filter_=['physics'])
			nt.show('test.html')

		#Network(notebook=True)
		st.subheader('Pyvis with Networkx for sentences Relationships')

		physics=st.checkbox('Add Physics Interactivity', value=True)
		simple_func(physics)

		HtmlFile = open("test.html", 'r', encoding='utf-8')
		source_code = HtmlFile.read() 
		components.html(source_code, height = 900,width=900)

		# Finished 
		st.success('This is a success message!', icon="✅")

if __name__ == '__main__':
	main()
  
