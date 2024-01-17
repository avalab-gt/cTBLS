# Ablation studies for cTBLS
import utils
from data_api import HybridDialogueDataset, get_hash
from transformers import RobertaTokenizer, RobertaModel
import pickle
from torch.utils.data import Dataset, DataLoader 
import torch
import torch.nn as nn 
from torch.optim import Adam, AdamW
from transformers import get_scheduler
from models import PQCLR, PQNCLR, PQNTriplet, PQNTriplet_Distributed
import numpy as np
import os
import openai
import json
import pandas as pd
from transformers import GPT2TokenizerFast
# tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
from nltk.corpus import stopwords
import tqdm
from rank_bm25 import BM25Okapi
from functools import lru_cache
import time 




class Passage_Positive_Anchors_Dataset(Dataset):
    def __init__(self, positive_embeddings, anchor_embeddings):
        self.positive_embeddings = positive_embeddings
        self.anchor_embeddings = anchor_embeddings

    def __len__(self):
        return len(self.positive_embeddings)
    
    def __getitem__(self, idx):
        return self.positive_embeddings[idx].clone().detach(), self.anchor_embeddings[idx].clone().detach()


"""
@lru_cache(maxsize=None)

def generate_table_embeddings(mode='validate'):
    # Generating table embeddings
    print("generating table embeddings")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = HybridDialogueDataset()
    ds_candidates = dataset.get_all_candidates()
    val_conversations = dataset.get_conversations(mode=mode)
    val_turn_ids = dataset.get_turn_ids(mode=mode)
    val_turns = dataset.get_turns(mode=mode)
    val_positives = []
    val_anchors = []
    val_sources = []
    evaluated_val_conversations = []

    top_level_info_df = utils.create_table_top_level_info(dataset) # pandas df containing tables and their corresponding information

    for val_turn_id in tqdm.tqdm(val_turn_ids, desc='Generating pairs'):
        val_turn = dataset.get_turn(val_turn_id)
        if val_turn['conversation_id'] in evaluated_val_conversations:
            continue # Only looking at the first turns 
        
        evaluated_val_conversations.append(val_turn['conversation_id'])
        val_query = val_turn['current_query']

        correct_candidate = ds_candidates[val_turn['correct_next_cands_ids'][0]]

        correct_source = correct_candidate['page_key'] or correct_candidate['table_key'].rsplit('_', 1)[0]
        val_sources.append(correct_candidate['page_key'] or correct_candidate['table_key'])
        correct_source = correct_source.replace("_", ' ').lower()

        possible_candidates = top_level_info_df[top_level_info_df['titles']==correct_source]
        if len(possible_candidates) > 1: # Multiple tables for this question exist, need to select the correct one
            if correct_candidate['table_key'] is not None:
                correct_id = int(correct_candidate['table_key'].rsplit('_', 1)[1])
                correct_info = list(possible_candidates[possible_candidates['id']==correct_id]['info'])[0]
            else:
                correct_info = possible_candidates.iloc[0]['info']
        else:
            correct_info = possible_candidates.iloc[0]['info']

        val_positives.append(correct_info)
        val_anchors.append(val_query)
    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    val_positive_encodings = tokenizer(val_positives, padding=True, truncation=True, return_tensors='pt')['input_ids']
    val_anchor_encodings = tokenizer(val_anchors, padding=True, truncation=True, return_tensors='pt')['input_ids']
    val_passage_dataset = Passage_Positive_Anchors_Dataset(positive_embeddings=val_positive_encodings, anchor_embeddings=val_anchor_encodings)

    val_dataloader = DataLoader(val_passage_dataset, batch_size=len(val_passage_dataset), shuffle=False)
    
    if torch.cuda.is_available():
        model = torch.load('fine-tuned-encoder.pt').to(device)
    else:
        model = torch.load('fine-tuned-encoder.pt', map_location=torch.device('cpu'))
    model = model.eval()

    for i,batch in enumerate(val_dataloader):
        with torch.no_grad():
            table_embeddings, _ = model(batch[0].clone().detach().to(device), batch[0].clone().detach().to(device))

    return table_embeddings
"""

def generate_table_embeddings(mode='test'):
    table_embeddings = torch.load('test_table_embeddings.pt')
    return table_embeddings



# Dense Table retrieval 
def dense_table_retrieval(query, mode, sources):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    if torch.cuda.is_available():
        model = torch.load('fine-tuned-encoder.pt').to(device)
    else:
        model = torch.load('fine-tuned-encoder.pt', map_location=torch.device('cpu'))

    text_encoding = tokenizer(query, padding='max_length', truncation=True, max_length=512, return_tensors='pt')['input_ids']
    text_encoding = text_encoding.to(device)

    model = model.to(device)
    model = model.eval()

    val_passage_dataset = Passage_Positive_Anchors_Dataset(positive_embeddings=text_encoding, anchor_embeddings=text_encoding)
    val_dataloader = DataLoader(val_passage_dataset, batch_size=1, shuffle=True)

    # if mode == 'validate':
    #     with open('val_sources_retriever.pickle', 'rb') as f:
    #         val_sources = pickle.load(f)

    # if torch.cuda.is_available():
        # pos_emb = torch.load('table_embeddings_retriever.pt') 
    pos_emb = generate_table_embeddings(mode=mode)
    # else:
    #     pos_emb = torch.load('table_embeddings_retriever.pt', map_location=torch.device('cpu'))

    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            _,text_embedding = model(batch[1].clone().detach().to(device), batch[1].clone().detach().to(device))
        

    output = torch.matmul(text_embedding, pos_emb.T)
    position=torch.argmax(output, dim=1).item()

    retrieved_source = sources[position]
    return retrieved_source




def knowledge_retrieval(query, candidates, top_3=True):
    max_len_tokenizer = 512

    # Compute Embeddings using Roberta
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    tokenizer.model_max_length = max_len_tokenizer 
    # device = 'cuda' if torch.cuda.is_available else 'cpu'
    device = 'cpu'
    model = PQNTriplet_Distributed(device=device)
    model_path = 'fine-tuned-qa_retriever_distributed_epoch_13_Jan_10.pt'
    # model.load_state_dict(torch.load(f'{model_path}'))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model = model.eval()


    anch_enc = tokenizer(query, padding="max_length", truncation=True, max_length=max_len_tokenizer, return_tensors='pt')['input_ids'].to(device)
    pos_enc = anch_enc 

    negative_dists = []

    with torch.no_grad():
        
        pos_emb_val, anch_emb_val, _ = model(pos_enc, anch_enc, torch.zeros_like(pos_enc).to(device))
        anch_emb_val = torch.nn.functional.normalize(anch_emb_val, dim=1)
        
        for i,candidate in enumerate(candidates):
            neg_enc = tokenizer(candidate, padding="max_length", truncation=True, max_length=max_len_tokenizer, return_tensors='pt')['input_ids'].to(device)
            neg_emb_val = model.passage_encoder(neg_enc).last_hidden_state
            neg_emb_val = torch.mean(neg_emb_val, dim=1)
            neg_emb_val = torch.nn.functional.normalize(neg_emb_val, dim=1)

            negative_dist = torch.matmul(neg_emb_val, anch_emb_val.T)
            negative_dist = negative_dist.squeeze()
            negative_dists.append(negative_dist.cpu().item())
        
        if type(negative_dists) != list:
            negative_dists = [negative_dists]

        if top_3: 
            max_neg_indices = np.argsort(negative_dists)[::-1][:3]
            max_neg_item = ''
            for index in max_neg_indices:
                max_neg_item +=candidates[index] + ' '    

            max_neg_item = max_neg_item.strip() # Remove trailing space 

        else: # Use top-1 knowledge 
            max_neg_index = np.argmax(negative_dists)
            max_neg_item = candidates[max_neg_index]

    print("max neg item", max_neg_item)

    return max_neg_item 



def response_generation(knowledge, query):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    s=set(stopwords.words('english'))
    head_prompt=f"Use the context and knowledge to generate a natural language response."

    add_string = f'\nContext: {query} \nKnowledge: {knowledge} \nResponse:'
    final_prompt = head_prompt + add_string

        
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=final_prompt,
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )

    nl_resp = response['choices'][0]['text']

    return nl_resp




def table_func(question):
    dataset = HybridDialogueDataset()
    mode = 'test'
    conversations = dataset.get_conversations(mode=mode)
    ds_candidates = dataset.get_all_candidates()
    turns = dataset.get_turns(mode=mode)

    table_cell_map = utils.get_table_cells_from_first_turn(dataset, tokenizer=None, mode=mode)
    sources = list(table_cell_map['sources'])


    validation_data = pd.DataFrame()
    gpt_responses = []
    gold_responses = []

    # DTR vs BM25
    table_retrieval_method = 'DTR' 
    knowledge_retrieval_top = 'top_3'

    # query = turns[turn_key]['current_query']
    query = question
    
    if table_retrieval_method == 'DTR':
        retrieved_source = dense_table_retrieval(query, mode=mode, sources=sources)

    print("Retrieved Source", retrieved_source)
    
    retrieved_source_index = sources.index(retrieved_source)

    candidates = table_cell_map.iloc[retrieved_source_index]['references']

    return retrieved_source, candidates
    """
    context = f"{query} {turns[turn_key]['long_response_to_query']}"
                # print(f"Q{i}: {query}")
                # print(f"A{i}: {turns[turn_key]['long_response_to_query']}")

            else:
                # Use the question to retrieve knowledge on the first turn
                turn_key = turn_keys[i]
                new_query = turns[turn_key]['current_query']

                context = f"{context} {new_query}"

                if knowledge_retrieval_top == 'top_3':
                    knowledge = knowledge_retrieval(context, candidates, top_3=True)
                elif knowledge_retrieval_top == 'top_1':
                    knowledge = knowledge_retrieval(context, candidates, top_3=False)
                gpt3_response = response_generation(knowledge, new_query)
                time.sleep(5) # Sleep for some time due to OpenAI Rate Limit Error
                gold_response = turns[turn_key]['long_response_to_query']

                context = f"{context} {gold_response}"
                # print(f"Q{i}: {new_query}")
                # print(f"A{i}: {gpt3_response}")

                datum = {'Gold_response': gold_response, 'GPT_response': gpt3_response}
                write_to_file(outfile, datum)
    """