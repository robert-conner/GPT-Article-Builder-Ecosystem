#import required libs
from tqdm import tqdm
import pickle
from pathlib import Path
import faiss
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.prompts import Prompt
from langchain.vectorstores import FAISS
import tiktoken
import os
import select
import subprocess
import sys
import time
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from docx import Document
import os

#To hide future warnings 
    #Deprecation warning pertain to future possible errors as the current code may be deprecated (eliminated)   
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 
import pandas as pd
#style options 

%matplotlib inline  
#if you want graphs to automatically without plt.show
pd.set_option('display.max_columns',500)

citations = {'Add Sources for Articles as a list in the same order as the documents in document_path'}

from docx import Document

def extract_content_from_docx(file_path):
    """Extract content from a Word document."""
    doc = Document(file_path)
    return [para.text for para in doc.paragraphs if len(para.text.strip()) > 0]

def is_unwanted_paragraph(para):
    # Define your criteria for an unwanted paragraph
    return False  # Placeholder, adjust as needed.

def generate_dataframe(docx_files, citation_mapping):
    """Generate a dataframe from extracted paragraphs and add corresponding citations."""
    data = []
    for file_path in docx_files:
        title = os.path.basename(file_path)
        citation = citation_mapping.get(title)
        paragraphs = extract_content_from_docx(file_path)
        for para in paragraphs:
            if not is_unwanted_paragraph(para):
                data.append([title, para, citation])
    return pd.DataFrame(data, columns=["document_id", "paragraph text", "citation"])

def get_all_docx_files(directory=".\Articles\Sources"):
    docx_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.docx'):
                docx_files.append(os.path.join(root, file))
    return docx_files

# Main Execution
docx_files = get_all_docx_files()
df = generate_dataframe(docx_files, citations)

# Define a function to split text based on common delimiters
def split_text(text):
    delimiters = ['\n', '\t', '\r']
    for delimiter in delimiters:
        text = text.replace(delimiter, '|SPLIT_HERE|')
    return text.split('|SPLIT_HERE|')

# Split the text based on delimiters and keep track of the original index
all_rows = []
for idx, row in df.iterrows():
    for segment in split_text(row['paragraph text']):
        new_row = row.copy()
        new_row['paragraph text'] = segment
        all_rows.append((idx, new_row))

# Sort rows by the original index and reset index
df = pd.DataFrame([row[1] for row in sorted(all_rows, key=lambda x: x[0])])
df = df.reset_index(drop=True)

#df.to_csv('text.csv')

# Check for duplicate paragraphs
df = df.drop_duplicates(subset='paragraph text', keep='first')
df = df.copy()
df['text_length'] = 0
df['text_length'] = df['paragraph text'].apply(len)

# List to hold dictionaries for new DataFrame
new_rows = []
buffer = None

# Function to check if a paragraph likely continues
def is_continuation(para):
    return para.endswith('-') or not para.endswith('.')

# Iterate over the original DataFrame
for i, row in df.iterrows():
    # Check if the current paragraph should be merged with the buffer
    if buffer is not None:
        # Check if the paragraph is a continuation or if the buffer paragraph is short
        if is_continuation(buffer['paragraph text']) or buffer['text_length'] < 200:
            buffer['paragraph text'] += ' ' + row['paragraph text']
            buffer['text_length'] += row['text_length']
        else:
            new_rows.append(buffer.to_dict())
            buffer = row  # Start a new buffer with the current row
    else:
        if is_continuation(row['paragraph text']) or row['text_length'] < 200:
            buffer = row
        else:
            new_rows.append(row.to_dict())

# Append the remaining buffer if not None
if buffer is not None:
    new_rows.append(buffer.to_dict())

# Create a new DataFrame from the list of dictionaries
merged_df = pd.DataFrame(new_rows)

merged_df['paragraph text'] = merged_df['paragraph text'].str.strip()

# Calculate mean and standard deviation
mean = merged_df['text_length'].mean()
std = merged_df['text_length'].std()

# Plotting the distribution
plt.figure(figsize=(10, 6))
plt.hist(merged_df['text_length'], bins=50, color='skyblue', edgecolor='black')
plt.title('Distribution of Modified Text Length')
plt.xlabel('Text Length')
plt.ylabel('Frequency')

# Drawing lines for standard deviations
plt.axvline(mean, color='red', linestyle='dashed', linewidth=2)  # Mean line
plt.axvline(mean + std, color='green', linestyle='dashed', linewidth=2)  # Mean + 1 STD
plt.axvline(mean - std, color='green', linestyle='dashed', linewidth=2)  # Mean - 1 STD
plt.axvline(mean + 2*std, color='yellow', linestyle='dashed', linewidth=2)  # Mean + 2 STD
plt.axvline(mean - 2*std, color='yellow', linestyle='dashed', linewidth=2)  # Mean - 2 STD
plt.axvline(mean + 3*std, color='blue', linestyle='dashed', linewidth=2)  # Mean + 3 STD
plt.axvline(mean - 3*std, color='blue', linestyle='dashed', linewidth=2)  # Mean - 3 STD

# Grid and layout
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

strings = merged_df['paragraph text'][merged_df['text_length']>2700].tolist()

for i in strings:
    # Count tokens
    token_count = num_tokens_from_string(i, "gpt-4")
    print(token_count)

def clean_text_series(text_series):
    return text_series.str.replace('\n', ' ').str.replace('\t', ' ').str.replace('\r', ' ').str.replace(' +', ' ').str.strip()
merged_df['paragraph text'] = clean_text_series(merged_df['paragraph text'])

display(merged_df)

os.environ["OPENAI_API_KEY"] = "INSERT YOUR OWN API KEY"

import time
from tqdm import tqdm  # If you can install packages, otherwise use a simple print loop.

def train(df):
    # Convert the 'paragraph text' column to a list
    docs = df['paragraph text'].tolist()

    store = FAISS.from_texts(docs, OpenAIEmbeddings())  
    faiss.write_index(store.index, "training.index")
    store.index = None
    
    print(f'\nTraining complete! Index has not been saved.\n')
    return store
      
def runBotChat(df, store):
    global history_pm
    global num_int

    index = faiss.read_index("training.index")

    store.index = index

    with open("training/instructions.txt", "r") as f:
        promptTemplate_assist = f.read()
    
    with open("training/instructions_pm.txt", "r") as f:
        promptTemplate = f.read()

    prompt = Prompt(template=promptTemplate_assist,
                    input_variables=["draft", "context", "question"])

    chatopenai = ChatOpenAI(temperature=0.75, model="gpt-4-1106-preview")
    llmChain = LLMChain(prompt=prompt, llm=chatopenai)

    prompt_pm = Prompt(template=promptTemplate, 
                    input_variables=["history", "question"])


    chatopenai_pm = ChatOpenAI(temperature=0.75, model="gpt-4-1106-preview")
    llmChain_pm = LLMChain(prompt=prompt_pm, llm=chatopenai_pm)
    
    def num_tokens_from_string(string: str, encoding_name: str) -> int:
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.encoding_for_model(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens
    
    def PM(subject, revision_flag = False, assist_ans=''):
        global history_pm
        global num_int
        
        if num_int == 0 and revision_flag==False:
            question_pm = f'Produce an indepth outline using SEO best practices for a blog article explaining {subject}. The outline should include seperate sections and a description of the type of information we need for the section. The introduction does not need to describe the purpose of the article, simply focus on the material.'
            print('Producing an outline for the article...')
            answer = llmChain_pm.predict(question=question_pm, history=history_pm)
            # Simulate processing time
            print(f"Task {num_int}: Article outline in progress, please wait...")
                    
            history_pm = history_pm + answer
            num_int = 1
            
            question_pm = f'Provide the first section and indepth section description based on the outline for our writing assistant to use to write the section. Provide only the section title and description. Do not include additional instructions.'

            answer = llmChain_pm.predict(question=question_pm, history=history_pm)

            # Simulate processing time
            print(f"Task {num_int}: Generating task, please wait...")
            print(history_pm)
            return answer
        elif num_int != 0 and revision_flag==False:
            question_pm = f'Provide the next section and section description based on the outline for our writing assistant. Provide only the section title and description. Do not include additional instructions.'

            answer = llmChain_pm.predict(question=question_pm, history=history_pm)

            # Simulate processing time
            print(f"Task {num_int}: Article generation in progress, please wait...")
            return answer

        if revision_flag == True:

            question_pm = f'Revise the this content for the latest section. Make sure we are not redundant or repetitive with our information between sections. Make sure the article has the appropriate flow. The provided citations should be added to our ongoing list of citations for the article and be correctly used within the text and referenced at the bottom of the article. Provide the section title followed by the revised content.\n\n{assist_ans}'

            answer = llmChain_pm.predict(question=question_pm, history=history_pm)

            # Simulate processing time
            print(f"Task {num_int}: Article generation in progress, please wait...")
            history_pm = history_pm + "\n\n" + answer
            return answer

    def onMessage(df, subject):
        global history_pm
        global num_int 
        
        question = PM(subject)
        docs_and_scores = store.similarity_search_with_score(question, k=20)

        added_paragraphs = set()
        context_texts = []
        source_list = []

        for doc, score in docs_and_scores:
            # Find the row in the original dataframe that matches the page_content of the Document
            matched_row = df[df['paragraph text'] == doc.page_content].iloc[0]
            matched_index = matched_row.name
            matched_paper_title = matched_row['document_id']
            matched_paper_citation = matched_row['citation']
            matched_paragraph_length = len(matched_row['paragraph text'])

            # Check if the matched paragraph is not already added
            if matched_index not in added_paragraphs:
                added_paragraphs.add(matched_index)

                # Extract context for the matched paragraph
                context_indices = [matched_index]  # Start with the matched index

                for offset in range(1, 2):  # Example: Check up to 2 paragraphs before and after
                    for idx in [matched_index - offset, matched_index + offset]:
                        if 0 <= idx < len(df) and df.loc[idx, 'document_id'] == matched_paper_title and idx not in added_paragraphs:
                            context_indices.append(idx)
                            added_paragraphs.add(idx)

                # Add the context paragraphs to context_texts
                for idx in context_indices:
                    paragraph_text = df.loc[idx, 'paragraph text']
                    context_texts.append(f'''####
                    Context from {matched_paper_citation}:
                    {paragraph_text}
                    
                    ''')

                    # Append data to context_data
                    source_list.append({
                        "document_id": matched_paper_title,
                        "citation": matched_paper_citation,
                        "paragraph_text": paragraph_text
                    })                    

        # Count tokens and generate answer
        token_count = num_tokens_from_string("".join(context_texts), "gpt-4")
        print(token_count)
        answer = llmChain.predict(draft=history_pm, context="".join(context_texts), question=question)
        
        print('Writing Assistant Answer')        
        print(answer)
        print('')
        print('')

        # Simulate processing time
        print("Tasked Writing Assistant: Article generation in progress, please wait...")
        num_int += 1
        pm_answer = PM(subject, revision_flag=True, assist_ans=answer)

        return pm_answer, source_list

   
    RED = "\033[91m"
    GREEN = "\033[92m"
    RESET = "\033[0m"

    while True:
        subject = input(GREEN + "Provide the subject (or type 'exit' to quit; 'continue' to continue) > " + RESET)
        if subject.lower() in ['exit', 'quit']:
            print("Exiting chat. Goodbye!")
            break
        answer, source_list = onMessage(df, subject)
        print(RED + f"Bot: {answer}" + RESET)
        print()
        
        # Convert source_list to a DataFrame
        source_list_df = pd.DataFrame(source_list)

        # Plot the histogram of citations
        plt.figure(figsize=(8, 4.5))
        sns.histplot(data=source_list_df, x='document_id', kde=True, color='blue')
        plt.xlabel('Document ID')
        plt.ylabel('Counts')
        plt.title('Histogram of Citations')
        plt.xticks(rotation=45, ha='center')
        plt.show()       

# Run Program
if "OPENAI_API_KEY" not in os.environ:
  print("You must set an OPENAI_API_KEY using the Secrets tool",
        file=sys.stderr)
else:

  print(">>>> Article Generator <<<<\n")
  print("Select an option:")
  print()
  print("1: Train Model\n2: Talk to your Bot\n3: Exit\n", end="")

global added_paragraphs
global history_pm
global num_int
    
exit_flag=True
while exit_flag:
    choice = input("Your choice: ").strip()
    if choice == "1":
        print("Launching bot training...")
        store = train(merged_df)
    elif choice == "2":
        print("Launching bot conversation mode...")

        history_pm = ''''''
        num_int = 0

        runBotChat(merged_df, store)
    elif choice == "3":
        print("Exiting, goodbye!")
        exit_flag = False
        break
