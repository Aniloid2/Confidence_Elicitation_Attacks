from textattack.goal_function_results import GoalFunctionResultStatus
from textattack.search_methods import SearchMethod
from textattack.shared import  WordEmbedding
from src.custom_constraints.sentence_encoders import UniversalSentenceEncoder
from textattack.shared.utils import device

import numpy as np
import torch
import nltk
import random
import math
import os
from collections import defaultdict
class TextHoaxerSearch(SearchMethod):
    def __init__(self, max_iterations=2,**kwargs): 
        for key, value in kwargs.items():
            setattr(self, key, value)

        nltk.download('averaged_perceptron_tagger')
        nltk.download('wordnet')

        self.max_iterations = max_iterations
        self.embedding = WordEmbedding.counterfitted_GLOVE_embedding() 
        self.sentence_encoder_use = UniversalSentenceEncoder(window_size=15)
        self.download_synonym_file()

        

        self.number_of_queries = 0

    def download_synonym_file(self):
        import gdown
        import pickle
        # Create the full directory path if it doesn't exist
        full_path = os.path.join(self.cache_dir, 'texthoaxer')
        os.makedirs(full_path, exist_ok=True)

        # Define the URL or Google Drive ID
        file_id = '1AIz8Imvv8OmHxVwY5kx10iwKAUzD6ODx'
        url = f'https://drive.google.com/uc?id={file_id}'

        # Define the file path where the file will be saved
        output_path = os.path.join(full_path, 'mat.txt')

        # Check if the file already exists
        if os.path.exists(output_path):
            print(f"File already exists at: {output_path}")
            self.ceattack_logger.warning(f'File already exists at: {output_path}')

        else:
            # Download the file
            print(f"Downloading file to: {output_path}")
            self.ceattack_logger.info(f'Downloading file to: {output_path}')

            gdown.download(url, output_path, quiet=False)
            print("Download complete.")
            self.ceattack_logger.info(f'Download complete.')

        
        with open(output_path, "rb") as fp:
            self.sim_lis = pickle.load(fp)
    def soft_threshold(self, alpha, beta):
        if beta > alpha:
            return beta - alpha
        elif beta < -alpha:
            return beta + alpha
        else:
            return 0
    def  get_pos(self,sent, tagset='universal'):
        '''
        :param sent: list of word strings
        tagset: {'universal', 'default'}
        :return: list of pos tags.
        Universal (Coarse) Pos tags has  12 categories
            - NOUN (nouns)
            - VERB (verbs)
            - ADJ (adjectives)
            - ADV (adverbs)
            - PRON (pronouns)
            - DET (determiners and articles)
            - ADP (prepositions and postpositions)
            - NUM (numerals)
            - CONJ (conjunctions)
            - PRT (particles)
            - . (punctuation marks)
            - X (a catch-all for other categories such as abbreviations or foreign words)
        '''
        if tagset == 'default':
            word_n_pos_list = nltk.pos_tag(sent)
        elif tagset == 'universal':
            word_n_pos_list = nltk.pos_tag(sent, tagset=tagset)
            
        _, pos_list = zip(*word_n_pos_list)
        return pos_list

    def perform_search(self, initial_result):
        self.number_of_queries = 0
        attacked_text = initial_result.attacked_text


        text_ls = attacked_text.words
        true_label = initial_result.ground_truth_output 
        orig_label = initial_result.output

        word_idx_dict = self.embedding._word2index
        embed_content = self.embedding
        idx2word = self.embedding._index2word
        word2idx = self.embedding._word2index
        criteria = self
        top_k_words = self.max_iter_i
        cos_sim = self.sim_lis
        budget =  self.query_budget - 1 # -1 because we have qrs > buget, but if we do a model call exacly on query results will be empty
        num_synonyms = self.n_embeddings
        pos_ls = criteria.get_pos(text_ls)
        len_text = len(text_ls)
        
        num_queries = 1
        rank = {}
        words_perturb = []
        
        
        pos_tags = nltk.pos_tag(text_ls)   
        
        for i, (word, pos_tag) in enumerate(pos_tags):
            if pos_tag.startswith(('VB', 'NN', 'JJ', 'RB')) and len(word) > 2:
                words_perturb.append((i, word))

        random.shuffle(words_perturb)
        words_perturb = words_perturb[:top_k_words]



        words_perturb_indices = [idx for idx, word in words_perturb]



        # Step 1: Initialization
        number_samples = self.num_transformations 
        self.number_of_queries+=number_samples + 1 # checking the original sample if it's correct, then num samples perturbations to find adv
        self.goal_function.num_queries = self.number_of_queries
        
        perturbed_text = [self.random_initialization(attacked_text,words_perturb_indices) for i in range(number_samples)]
        
        
        results, search_over = self.get_goal_results(perturbed_text)
        
         
         
        results_success=[]
        perturbed_text_success = []
        for i,result in enumerate(results):
            if result.ground_truth_output!=result.output:
                results_success.append(result)
                perturbed_text_success.append(perturbed_text[i])

 
        if len(results_success) == 0:
            flag = 0
        else:
            flag = 1

             
            perturbed_text = perturbed_text_success[0] 

            print (f'TextHoaxer Original Text: \n {attacked_text}')
            self.ceattack_logger.debug(f'TextHoaxer Original Text: \n {attacked_text}')
                
            print (f'TextHoaxer Perturbed Text: \n {perturbed_text}') 
            self.ceattack_logger.debug(f'TextHoaxer Perturbed Text: \n {perturbed_text}')

            random_text = perturbed_text.words



        words_perturb_idx= []
        words_perturb_embed = []
        words_perturb_doc_idx = []
        for idx, word in words_perturb:
            if word in word_idx_dict:
                words_perturb_doc_idx.append(idx)
                words_perturb_idx.append(word2idx[word])
                words_perturb_embed.append([float(num) for num in embed_content[ word_idx_dict[word] ]])
                 
        words_perturb_embed_matrix = np.asarray(words_perturb_embed)


        synonym_words,synonym_values=[],[]
        for idx in words_perturb_idx:
            res = list(zip(*(cos_sim[idx])))
            
             
            res[0] = res[0][:self.n_embeddings]
            res[1] = res[1][:self.n_embeddings]
             
            temp=[]
            for ii in res[1]:
                temp.append(idx2word[ii])
            
            
            synonym_words.append(temp)
            temp=[]
            for ii in res[0]:
                temp.append(ii)
            
            
            synonym_values.append(temp)

        synonyms_all = []
        synonyms_dict = defaultdict(list)
        for idx, word in words_perturb:
            if word in word2idx:
                synonyms = synonym_words.pop(0)
                if synonyms:
                    synonyms_all.append((idx, synonyms))
                    synonyms_dict[word] = synonyms

        
        
        qrs = number_samples + 1
        old_qrs = qrs  
        
        
            
            
        if flag == 1:
            changed = 0
            for i in range(len(text_ls)):
                if text_ls[i]!=random_text[i]:
                    changed+=1
             

             
            while True:
                choices = []

                for i in range(len(text_ls)):
                    if random_text[i] != text_ls[i]:
                        new_text = random_text[:]
                        new_text[i] = text_ls[i]
                         


                        new_text_joint = attacked_text.generate_new_attacked_text(new_text) 


                         
                        if attacked_text.text == new_text_joint.text: # is word sub leads to perturbation being same as original sample skip
                            continue
                        

                        sim_remove_unnecessary_org, sim_remove_unnecessary_pert = self.sentence_encoder_use.encode([attacked_text.text, new_text_joint.text])

                        if not isinstance(sim_remove_unnecessary_org, torch.Tensor):
                            sim_remove_unnecessary_org = torch.tensor(sim_remove_unnecessary_org)

                        if not isinstance(sim_remove_unnecessary_pert, torch.Tensor):
                            sim_remove_unnecessary_pert = torch.tensor(sim_remove_unnecessary_pert)

                        semantic_sims = self.sentence_encoder_use.sim_metric(sim_remove_unnecessary_org.unsqueeze(0), sim_remove_unnecessary_pert.unsqueeze(0))
                        
                         
                        results, search_over = self.get_goal_results([new_text_joint])
                        if search_over: 
                             
                            return initial_result
                         
                        qrs+=1
                        self.number_of_queries+=1
                        self.goal_function.num_queries = self.number_of_queries
                         
                        if results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                            choices.append((i,semantic_sims[0]))
 
                         

                 
                if len(choices) > 0:
                    choices.sort(key = lambda x: x[1])
                    choices.reverse()
                    for i in range(len(choices)):
                        new_text = random_text[:]
                        new_text[choices[i][0]] = text_ls[choices[i][0]]

                         

                        new_text_joint = attacked_text.generate_new_attacked_text(new_text)
                        if attacked_text.text == new_text_joint.text: # is word sub leads to perturbation being same as original sample skip
                            continue

                        results, search_over = self.get_goal_results([new_text_joint])
                        if search_over: 
                             
                            return initial_result

                        qrs+=1
                        self.number_of_queries+=1
                        self.goal_function.num_queries = self.number_of_queries
                         
                        if results[0].goal_status != GoalFunctionResultStatus.SUCCEEDED:
                            break
                        random_text[choices[i][0]] = text_ls[choices[i][0]]

                         

                if len(choices) == 0:
                    break
            
             
            print (f'TextHoaxer after choices : \n {random_text}') 
            self.ceattack_logger.debug(f'TextHoaxer after choices: \n {random_text}')

            changed_indices = [] 
            num_changed = 0
            for i in range(len(text_ls)):
                if text_ls[i]!=random_text[i]:
                    changed_indices.append(i)
                    num_changed+=1
             

            new_random_text_joint = attacked_text.generate_new_attacked_text(random_text) 
             
            
            sim_remove_unnecessary_org, sim_remove_unnecessary_pert = self.sentence_encoder_use.encode([attacked_text.text, new_random_text_joint.text])

            if not isinstance(sim_remove_unnecessary_org, torch.Tensor):
                sim_remove_unnecessary_org = torch.tensor(sim_remove_unnecessary_org)

            if not isinstance(sim_remove_unnecessary_pert, torch.Tensor):
                sim_remove_unnecessary_pert = torch.tensor(sim_remove_unnecessary_pert)

            random_sim = self.sentence_encoder_use.sim_metric(sim_remove_unnecessary_org.unsqueeze(0), sim_remove_unnecessary_pert.unsqueeze(0))
            random_sim = random_sim.item()
             
            if qrs > budget:
                 
                random_text_qrs_joint = attacked_text.generate_new_attacked_text(random_text) 

                sim_remove_unnecessary_org, sim_remove_unnecessary_pert = self.sentence_encoder_use.encode([attacked_text.text, random_text_qrs_joint.text])

                if not isinstance(sim_remove_unnecessary_org, torch.Tensor):
                    sim_remove_unnecessary_org = torch.tensor(sim_remove_unnecessary_org)

                if not isinstance(sim_remove_unnecessary_pert, torch.Tensor):
                    sim_remove_unnecessary_pert = torch.tensor(sim_remove_unnecessary_pert)

                failed_sem_sim = self.sentence_encoder_use.sim_metric(sim_remove_unnecessary_org.unsqueeze(0), sim_remove_unnecessary_pert.unsqueeze(0))
                failed_sem_sim = failed_sem_sim.item()
                 

                if failed_sem_sim <  (1 - (self.similarity_threshold) / math.pi): 
                    print (f'TextHoaxer best semantic similarity {failed_sem_sim} is too low: \n {random_text_qrs_joint}') 
                    self.ceattack_logger.debug(f'TextHoaxer best semantic similarity {failed_sem_sim} is too low: \n {random_text_qrs_joint}')
                    return initial_result
                else:
                    print (f'TextHoaxer run out of queries for sample: \n {random_text_qrs_joint}') 
                    self.ceattack_logger.debug(f'TextHoaxer run out of queries for sample: \n {random_text_qrs_joint}')
                    results_inner, search_over = self.get_goal_results([random_text_qrs_joint])
                     
                    return results_inner[0]
                
                

       
            if num_changed == 1:
                random_text_num_changed_joint = attacked_text.generate_new_attacked_text(random_text)  
                sim_remove_unnecessary_org, sim_remove_unnecessary_pert = self.sentence_encoder_use.encode([attacked_text.text, random_text_num_changed_joint.text])

                if not isinstance(sim_remove_unnecessary_org, torch.Tensor):
                    sim_remove_unnecessary_org = torch.tensor(sim_remove_unnecessary_org)

                if not isinstance(sim_remove_unnecessary_pert, torch.Tensor):
                    sim_remove_unnecessary_pert = torch.tensor(sim_remove_unnecessary_pert)

                failed_sem_sim = self.sentence_encoder_use.sim_metric(sim_remove_unnecessary_org.unsqueeze(0), sim_remove_unnecessary_pert.unsqueeze(0))
                failed_sem_sim = failed_sem_sim.item() 

                if failed_sem_sim <  (1 - (self.similarity_threshold) / math.pi):
                    print (f'TextHoaxer best semantic similarity {failed_sem_sim} is too low: \n {random_text_num_changed_joint}') 
                    self.ceattack_logger.debug(f'TextHoaxer best semantic similarity {failed_sem_sim} is too low: \n {random_text_num_changed_joint}')
                    
                    return initial_result
                else:
                    print (f'TextHoaxer run out of queries for sample: \n {random_text_num_changed_joint}') 
                    self.ceattack_logger.debug(f'TextHoaxer run out of queries for sample: \n {random_text_num_changed_joint}')
                     
                    results_inner, search_over = self.get_goal_results([random_text_num_changed_joint]) 
                    return results_inner[0]
                    
                 

            best_attack = random_text 
            best_sim = random_sim

            already_explored = set()


            gamma = 0.3*np.ones([words_perturb_embed_matrix.shape[0], 1])
            l1 = 0.1
            l2_lambda = 0.1
            for t in range(100):

                theta_old_text = best_attack
                sim_old= best_sim 
                old_adv_embed = []
                for idx in words_perturb_doc_idx:
                     
                    old_adv_embed.append([float(num) for num in embed_content[word_idx_dict[theta_old_text[idx]]]])
                old_adv_embed_matrix = np.asarray(old_adv_embed)

                theta_old = old_adv_embed_matrix-words_perturb_embed_matrix
               
                u_vec = np.random.normal(loc=0.0, scale=1,size=theta_old.shape)
                theta_old_neighbor = theta_old+0.5*u_vec
                 
                # Check if theta_old_neighbor is a 2D array
                if theta_old_neighbor.ndim != 2:
                    print('Theta_old_neighbor not a 2D array. Skipping this iteration.') 
                    self.ceattack_logger.debug(f'Theta_old_neighbor not a 2D array. Skipping this iteration.')
                     
                    continue
                theta_perturb_dist = np.sum((theta_old_neighbor)**2, axis=1)
                nonzero_ele = np.nonzero(np.linalg.norm(theta_old,axis = -1))[0].tolist()
                perturb_strength_order = np.argsort(-theta_perturb_dist[nonzero_ele])

                theta_old_neighbor_text = text_ls[:]
                for perturb_idx in range(len(nonzero_ele)):
                    perturb_word_idx = nonzero_ele[perturb_strength_order[perturb_idx]]
                    word_dict_idx = words_perturb_idx[perturb_word_idx]
                    
                    perturb_target = words_perturb_embed_matrix[perturb_word_idx]+theta_old_neighbor[perturb_word_idx]
                    syn_feat_set = []
                    for syn in synonyms_all[perturb_word_idx][1]:
                        
                        syn_feat = [float(num) for num in embed_content[word_idx_dict[syn]]]
                        syn_feat_set.append(syn_feat)

                    perturb_syn_dist = np.sum((syn_feat_set-perturb_target)**2, axis=1)
                    perturb_syn_order = np.argsort(perturb_syn_dist)
                    replacement = synonyms_all[perturb_word_idx][1][perturb_syn_order[0]]
                    
                    theta_old_neighbor_text[synonyms_all[perturb_word_idx][0]] = replacement

                    theta_old_neighbor_text_joint = attacked_text.generate_new_attacked_text(theta_old_neighbor_text)
                    
                    print (f'TextHoaxer after theta old neighbor : \n {theta_old_neighbor_text_joint}') 
                    self.ceattack_logger.debug(f'TextHoaxer after theta old neighbor: \n {theta_old_neighbor_text_joint}')

                    
                    
                    if attacked_text.text == theta_old_neighbor_text_joint.text: # is word sub leads to perturbation being same as original sample skip
                        continue 
                     
                    results, search_over = self.get_goal_results([theta_old_neighbor_text_joint])
                    if search_over: 
                         
                        return initial_result
                     
                    qrs+=1
                    self.number_of_queries+=1
                    self.goal_function.num_queries = self.number_of_queries
                     
                    
                    if qrs > budget:
                        sim_remove_unnecessary_org, sim_remove_unnecessary_pert = self.sentence_encoder_use.encode([attacked_text.text, theta_old_neighbor_text_joint.text])

                        if not isinstance(sim_remove_unnecessary_org, torch.Tensor):
                            sim_remove_unnecessary_org = torch.tensor(sim_remove_unnecessary_org)

                        if not isinstance(sim_remove_unnecessary_pert, torch.Tensor):
                            sim_remove_unnecessary_pert = torch.tensor(sim_remove_unnecessary_pert)

                        failed_sem_sim = self.sentence_encoder_use.sim_metric(sim_remove_unnecessary_org.unsqueeze(0), sim_remove_unnecessary_pert.unsqueeze(0))
                        failed_sem_sim = failed_sem_sim.item()
                         

                        if failed_sem_sim <  (1 - (self.similarity_threshold) / math.pi):  
                            print (f'TextHoaxer best semantic similarity {failed_sem_sim} is too low: \n {theta_old_neighbor_text_joint}') 
                            self.ceattack_logger.debug(f'TextHoaxer best semantic similarity {failed_sem_sim} is too low: \n {theta_old_neighbor_text_joint}')
                    
                            return initial_result
                        else:
                            print (f'TextHoaxer run out of queries for sample: \n {theta_old_neighbor_text_joint}') 
                            self.ceattack_logger.debug(f'TextHoaxer run out of queries for sample: \n {theta_old_neighbor_text_joint}')
                             
                            return results[0]
                        
                        
 
                    if results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                        break

 
                if results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:

                     
                    theta_old_neighbor_text_joint = attacked_text.generate_new_attacked_text(theta_old_neighbor_text) 
                    
                     
                    if attacked_text.text == theta_old_neighbor_text_joint.text: # is word sub leads to perturbation being same as original sample skip
                        continue 
                     
                    sim_remove_unnecessary_org, sim_remove_unnecessary_pert = self.sentence_encoder_use.encode([attacked_text.text, theta_old_neighbor_text_joint.text])

                    if not isinstance(sim_remove_unnecessary_org, torch.Tensor):
                        sim_remove_unnecessary_org = torch.tensor(sim_remove_unnecessary_org)

                    if not isinstance(sim_remove_unnecessary_pert, torch.Tensor):
                        sim_remove_unnecessary_pert = torch.tensor(sim_remove_unnecessary_pert)

                    sim_new = self.sentence_encoder_use.sim_metric(sim_remove_unnecessary_org.unsqueeze(0), sim_remove_unnecessary_pert.unsqueeze(0))
                    sim_new = sim_new.item() 

                    
                    derivative = (sim_old-sim_new)/0.5

                    g_hat = derivative*u_vec

                    theta_new = theta_old-0.3*(g_hat+2*l2_lambda*theta_old)

                    if sim_new > sim_old:
                        best_attack = theta_old_neighbor_text
                        best_sim = sim_new

                    theta_perturb_dist = np.sum((theta_new)**2, axis=1)
                    nonzero_ele = np.nonzero(np.linalg.norm(theta_new,axis = -1))[0].tolist()
                    perturb_strength_order = np.argsort(-theta_perturb_dist[nonzero_ele])

                    theta_new_text = text_ls[:]
                    for perturb_idx in range(len(nonzero_ele)):
                        perturb_word_idx = nonzero_ele[perturb_strength_order[perturb_idx]]
                        word_dict_idx = words_perturb_idx[perturb_word_idx]
                        
                        perturb_target = words_perturb_embed_matrix[perturb_word_idx]+theta_new[perturb_word_idx]
                        syn_feat_set = []
                        for syn in synonyms_all[perturb_word_idx][1]:
                            syn_feat = [float(num) for num in embed_content[word_idx_dict[syn]]]
                            syn_feat_set.append(syn_feat)

                        perturb_syn_dist = np.sum((syn_feat_set-perturb_target)**2, axis=1)
                        perturb_syn_order = np.argsort(perturb_syn_dist)
                        replacement = synonyms_all[perturb_word_idx][1][perturb_syn_order[0]]
                        
                        theta_new_text[synonyms_all[perturb_word_idx][0]] = replacement


                        theta_new_text_joint = attacked_text.generate_new_attacked_text(theta_new_text)
                        print (f'TextHoaxer after theta new : \n {theta_new_text_joint}') 
                        self.ceattack_logger.debug(f'TextHoaxer after theta new: \n {theta_new_text_joint}')
 
                        if attacked_text.text == theta_new_text_joint.text: # is word sub leads to perturbation being same as original sample skip
                            continue  
                         
                        results, search_over = self.get_goal_results([theta_new_text_joint])
                        if search_over: 
                             
                            return initial_result

                        qrs+=1
                        self.number_of_queries+=1
                        self.goal_function.num_queries = self.number_of_queries
                        
                         
                        if qrs > budget:
                            sim_remove_unnecessary_org, sim_remove_unnecessary_pert = self.sentence_encoder_use.encode([attacked_text.text, theta_new_text_joint.text])

                            if not isinstance(sim_remove_unnecessary_org, torch.Tensor):
                                sim_remove_unnecessary_org = torch.tensor(sim_remove_unnecessary_org)

                            if not isinstance(sim_remove_unnecessary_pert, torch.Tensor):
                                sim_remove_unnecessary_pert = torch.tensor(sim_remove_unnecessary_pert)

                            failed_sem_sim = self.sentence_encoder_use.sim_metric(sim_remove_unnecessary_org.unsqueeze(0), sim_remove_unnecessary_pert.unsqueeze(0))
                            failed_sem_sim = failed_sem_sim.item() 

                            if failed_sem_sim <  (1 - (self.similarity_threshold) / math.pi): 
                                print (f'TextHoaxer best semantic similarity {failed_sem_sim} is too low: \n {theta_new_text_joint}') 
                                self.ceattack_logger.debug(f'TextHoaxer best semantic similarity {failed_sem_sim} is too low: \n {theta_new_text_joint}')
                    
                                return initial_result
                            else: 
                                print (f'TextHoaxer run out of queries for sample: \n {theta_new_text_joint}') 
                                self.ceattack_logger.debug(f'TextHoaxer run out of queries for sample: \n {theta_new_text_joint}')
                                
                                return results[0]
              
              
                        if results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                            break



                    if results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                        theta_new_text_joint = attacked_text.generate_new_attacked_text(theta_new_text) 
                         
                        if attacked_text.text == theta_new_text_joint.text: # is word sub leads to perturbation being same as original sample skip
                            continue
                         
                        sim_remove_unnecessary_org, sim_remove_unnecessary_pert = self.sentence_encoder_use.encode([attacked_text.text, theta_new_text_joint.text])

                        if not isinstance(sim_remove_unnecessary_org, torch.Tensor):
                            sim_remove_unnecessary_org = torch.tensor(sim_remove_unnecessary_org)

                        if not isinstance(sim_remove_unnecessary_pert, torch.Tensor):
                            sim_remove_unnecessary_pert = torch.tensor(sim_remove_unnecessary_pert)

                        sim_theta_new = self.sentence_encoder_use.sim_metric(sim_remove_unnecessary_org.unsqueeze(0), sim_remove_unnecessary_pert.unsqueeze(0))
                        sim_theta_new = sim_theta_new.item()
                        
                        
                        if sim_theta_new > best_sim:
                            best_attack = theta_new_text
                            best_sim = sim_theta_new

                    
                    
                    
                    if results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                        gamma_old_text = theta_new_text

                        gamma_old_text_joint = attacked_text.generate_new_attacked_text(gamma_old_text) 
                        
                        print (f'TextHoaxer after gamma old : \n {gamma_old_text_joint}') 
                        self.ceattack_logger.debug(f'TextHoaxer after gamma old: \n {gamma_old_text_joint}')
 
                        if attacked_text.text == gamma_old_text_joint.text: # is word sub leads to perturbation being same as original sample skip
                            continue
                        
                        sim_remove_unnecessary_org, sim_remove_unnecessary_pert = self.sentence_encoder_use.encode([attacked_text.text, gamma_old_text_joint.text])

                        if not isinstance(sim_remove_unnecessary_org, torch.Tensor):
                            sim_remove_unnecessary_org = torch.tensor(sim_remove_unnecessary_org)

                        if not isinstance(sim_remove_unnecessary_pert, torch.Tensor):
                            sim_remove_unnecessary_pert = torch.tensor(sim_remove_unnecessary_pert)

                        gamma_sim_full = self.sentence_encoder_use.sim_metric(sim_remove_unnecessary_org.unsqueeze(0), sim_remove_unnecessary_pert.unsqueeze(0))
                        gamma_sim_full = gamma_sim_full.item()
                        
                        
                        gamma_old_adv_embed = []
                        for idx in words_perturb_doc_idx:
                             
                            gamma_old_adv_embed.append([float(num) for num in embed_content[word_idx_dict[gamma_old_text[idx]]]])
                        gamma_old_adv_embed_matrix = np.asarray(gamma_old_adv_embed)

                        gamma_old_pert= gamma_old_adv_embed_matrix-words_perturb_embed_matrix
                        gamma_old_pert_divided =gamma_old_pert/gamma
                        perturb_gradient = []
                        for i in range(gamma.shape[0]):
                            idx = words_perturb_doc_idx[i]
                            replaceback_text = gamma_old_text[:]
                            replaceback_text[idx] = text_ls[idx]

                            replaceback_text_joint = attacked_text.generate_new_attacked_text(replaceback_text) 
                            
                            print (f'TextHoaxer after replaced back : \n {replaceback_text_joint}') 
                            self.ceattack_logger.debug(f'TextHoaxer after replaced back: \n {replaceback_text_joint}')
    
                            if attacked_text.text == replaceback_text_joint.text: # is word sub leads to perturbation being same as original sample skip
                                continue
                            
                            sim_remove_unnecessary_org, sim_remove_unnecessary_pert = self.sentence_encoder_use.encode([attacked_text.text, replaceback_text_joint.text])

                            if not isinstance(sim_remove_unnecessary_org, torch.Tensor):
                                sim_remove_unnecessary_org = torch.tensor(sim_remove_unnecessary_org)

                            if not isinstance(sim_remove_unnecessary_pert, torch.Tensor):
                                sim_remove_unnecessary_pert = torch.tensor(sim_remove_unnecessary_pert)

                            replaceback_sims = self.sentence_encoder_use.sim_metric(sim_remove_unnecessary_org.unsqueeze(0), sim_remove_unnecessary_pert.unsqueeze(0))
                            replaceback_sims = replaceback_sims.item()
                            
                            
                            gradient_2 = self.soft_threshold(l1,gamma[i][0])
                            gradient_1 = -((gamma_sim_full-replaceback_sims)/(gamma[i]+1e-4))[0]
                            gradient = gradient_1+gradient_2
                            gamma[i]=gamma[i]-0.05*gradient


                        theta_new = gamma_old_pert_divided * gamma
                        theta_perturb_dist = np.sum((theta_new)**2, axis=1)
                        nonzero_ele = np.nonzero(np.linalg.norm(theta_new,axis = -1))[0].tolist()
                        perturb_strength_order = np.argsort(-theta_perturb_dist[nonzero_ele])
                        theta_new_text = text_ls[:]
                        for perturb_idx in range(len(nonzero_ele)):
                            perturb_word_idx = nonzero_ele[perturb_strength_order[perturb_idx]]
                            word_dict_idx = words_perturb_idx[perturb_word_idx]
                            
                            perturb_target = words_perturb_embed_matrix[perturb_word_idx]+theta_new[perturb_word_idx]
                            syn_feat_set = []
                            for syn in synonyms_all[perturb_word_idx][1]:
                                
                                syn_feat = [float(num) for num in embed_content[word_idx_dict[syn]]]
                                syn_feat_set.append(syn_feat)

                            perturb_syn_dist = np.sum((syn_feat_set-perturb_target)**2, axis=1)
                            perturb_syn_order = np.argsort(perturb_syn_dist)
                            replacement = synonyms_all[perturb_word_idx][1][perturb_syn_order[0]]
                            
                            theta_new_text[synonyms_all[perturb_word_idx][0]] = replacement


                            theta_new_text_joint = attacked_text.generate_new_attacked_text(theta_new_text) 
                            print (f'TextHoaxer after theta new 2 : \n {theta_new_text_joint}') 
                            self.ceattack_logger.debug(f'TextHoaxer after theta new 2: \n {theta_new_text_joint}')
    
                            if attacked_text.text == theta_new_text_joint.text: # is word sub leads to perturbation being same as original sample skip
                                continue 
                            
                            
                            results, search_over = self.get_goal_results([theta_new_text_joint])
                            if search_over:
                                
                                return initial_result

                            qrs+=1 
                            self.number_of_queries+=1
                            self.goal_function.num_queries = self.number_of_queries
                            
                            
                            
                            if qrs > budget:
                                sim_remove_unnecessary_org, sim_remove_unnecessary_pert = self.sentence_encoder_use.encode([attacked_text.text, theta_new_text_joint.text])

                                if not isinstance(sim_remove_unnecessary_org, torch.Tensor):
                                    sim_remove_unnecessary_org = torch.tensor(sim_remove_unnecessary_org)

                                if not isinstance(sim_remove_unnecessary_pert, torch.Tensor):
                                    sim_remove_unnecessary_pert = torch.tensor(sim_remove_unnecessary_pert)

                                failed_sem_sim = self.sentence_encoder_use.sim_metric(sim_remove_unnecessary_org.unsqueeze(0), sim_remove_unnecessary_pert.unsqueeze(0))
                                failed_sem_sim = failed_sem_sim.item()
                                

                                if failed_sem_sim <  (1 - (self.similarity_threshold) / math.pi):
                                    
                                    print (f'TextHoaxer best semantic similarity {failed_sem_sim} is too low: \n {theta_new_text_joint}') 
                                    self.ceattack_logger.debug(f'TextHoaxer best semantic similarity {failed_sem_sim} is too low: \n {theta_new_text_joint}')
                        
                                    
                                    return initial_result
                                else: 
                                    print (f'TextHoaxer run out of queries for sample: \n {theta_new_text_joint}') 
                                    self.ceattack_logger.debug(f'TextHoaxer run out of queries for sample: \n {theta_new_text_joint}')
                                
                                    return results[0]
                                
                            if results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                                break

                            
                            
                        if results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                            theta_new_text_joint = attacked_text.generate_new_attacked_text(theta_new_text) 
                            
                            
                            print (f'TextHoaxer after theta new 2: \n {theta_new_text_joint}') 
                            self.ceattack_logger.debug(f'TextHoaxer after theta new 2: \n {theta_new_text_joint}')
    
                            if attacked_text.text == theta_new_text_joint.text: # is word sub leads to perturbation being same as original sample skip
                                continue
                            
                            sim_remove_unnecessary_org, sim_remove_unnecessary_pert = self.sentence_encoder_use.encode([attacked_text.text, theta_new_text_joint.text])

                            if not isinstance(sim_remove_unnecessary_org, torch.Tensor):
                                sim_remove_unnecessary_org = torch.tensor(sim_remove_unnecessary_org)

                            if not isinstance(sim_remove_unnecessary_pert, torch.Tensor):
                                sim_remove_unnecessary_pert = torch.tensor(sim_remove_unnecessary_pert)

                            sim_theta_new = self.sentence_encoder_use.sim_metric(sim_remove_unnecessary_org.unsqueeze(0), sim_remove_unnecessary_pert.unsqueeze(0))
                            sim_theta_new = sim_theta_new.item()
                            
                            if sim_theta_new > best_sim:
                                best_attack = theta_new_text
                                best_sim = sim_theta_new


            best_attack_joint = attacked_text.generate_new_attacked_text(best_attack)
            

 
            print (f'TextHoaxer best attack: \n {best_attack_joint}') 
            self.ceattack_logger.debug(f'TextHoaxer best attack: \n {best_attack_joint}')

            if attacked_text.text == best_attack_joint.text: # is word sub leads to perturbation being same as original sample skip
                
                return initial_result # in this case i return the initial result, shouldent really ever happen since we always check if the perturbed sample is the same as original one at each step and ignore perturbation if it is.
            
            sim_remove_unnecessary_org, sim_remove_unnecessary_pert = self.sentence_encoder_use.encode([attacked_text.text, best_attack_joint.text])

            if not isinstance(sim_remove_unnecessary_org, torch.Tensor):
                sim_remove_unnecessary_org = torch.tensor(sim_remove_unnecessary_org)

            if not isinstance(sim_remove_unnecessary_pert, torch.Tensor):
                sim_remove_unnecessary_pert = torch.tensor(sim_remove_unnecessary_pert)

            best_sem_sim = self.sentence_encoder_use.sim_metric(sim_remove_unnecessary_org.unsqueeze(0), sim_remove_unnecessary_pert.unsqueeze(0))
            best_sem_sim = best_sem_sim.item()
            

            if best_sem_sim <  (1 - (self.similarity_threshold) / math.pi): 
                print (f'TextHoaxer best semantic similarity {best_sem_sim} is too low: \n {best_attack_joint}') 
                self.ceattack_logger.debug(f'TextHoaxer best semantic similarity {best_sem_sim} is too low: \n {best_attack_joint}')
                        
                
                return initial_result


            results, search_over = self.get_goal_results([best_attack_joint])
            self.number_of_queries+=1
            self.goal_function.num_queries = self.number_of_queries
            
            if search_over: 
                
                return initial_result
            
            print (f'TextHoaxer result best attack: \n {results}') 
            self.ceattack_logger.debug(f'TextHoaxer result best attack: \n {results}')
                 
            return results[0]


        else: 
            print (f'TextHoaxer no adversarial examples found with maximum number of samples args.num_transformations = {number_samples}, maybe try to increase this number? ') 
            self.ceattack_logger.debug(f'TextHoaxer no adversarial examples found with maximum number of samples args.num_transformations = {number_samples}, maybe try to increase this number? ')
               
            return initial_result
        


    def random_initialization(self, text, words_perturb):
        words = text.words
        tmp_text = text
        size_text = len(text.words)
        start_i = 0
        while start_i < size_text:
            # print ('start tmp text',tmp_text)
            if start_i not in words_perturb:
                start_i+=1
                continue
            
            words = tmp_text.words
            pos_tags = nltk.pos_tag(words)  
            
            if pos_tags[start_i][1].startswith(('VB', 'NN', 'JJ', 'RB')): 
                
                replaced_with_synonyms = self.get_transformations(tmp_text, original_text=tmp_text,indices_to_modify=[start_i])
                
                
                if replaced_with_synonyms:
                    tmp_text = random.choice(replaced_with_synonyms)
                else:
                    pass
                
            start_i+=1
            
        adv_text = tmp_text
        return adv_text


    def get_transformations(self, text, index):
        return self.transformation(text, index)

    def get_similarity(self, word1, word2):
        return self.transformation.get_cosine_similarity(word1, word2)
    
    @property
    def is_black_box(self):
        return True
