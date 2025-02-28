
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

class SSPAttackSearch(SearchMethod):
    def __init__(self, max_iterations=2,**kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        nltk.download('averaged_perceptron_tagger')
        nltk.download('wordnet')
        self.max_iterations = max_iterations
        self.embedding = WordEmbedding.counterfitted_GLOVE_embedding() 
        self.sentence_encoder_use = UniversalSentenceEncoder(window_size=15)
        
        self.number_of_queries = 0

    def check_model_status(self,input_text,check_skip=False):
        model_outputs = self.goal_function._call_model([input_text])
        current_goal_status = self.goal_function._get_goal_status(
            model_outputs[0], input_text, check_skip=check_skip
        )
        self.goal_function.num_queries +=1
        return current_goal_status

    def perform_search(self, initial_result):
        self.number_of_queries = 0
        attacked_text = initial_result.attacked_text

        
        
        # Step 1: Initialize sample
        number_samples = self.num_transformations
        
        perturbed_text = [self.random_initialization(attacked_text) for i in range(number_samples)]
        
        results, search_over = self.get_goal_results(perturbed_text) 
        self.number_of_queries+=number_samples + 1
        
        
        results_success = [result for result in results if result.ground_truth_output!=result.output] 

        results_success=[]
        perturbed_text_success = []
        for i,result in enumerate(results):
            if result.ground_truth_output!=result.output:
                results_success.append(result)
                perturbed_text_success.append(perturbed_text[i])

 
        if len(results_success) == 0:
            final_result = results[0] 
            return final_result # return a random result that wasnt perturbed to show it failed.

        # We use one sample to optimize for quality
        perturbed_text = perturbed_text_success[0]
        results = results_success[0]
 


        print (f'SSPAttack Original Text: \n {attacked_text}')
        self.ceattack_logger.debug(f'SSPAttack Original Text: \n {attacked_text}')
            
        print (f'SSPAttack Perturbed Text: \n {perturbed_text}') 
        self.ceattack_logger.debug(f'SSPAttack Perturbed Text: \n {perturbed_text}')
                 

        
         
        for _ in range(self.max_iterations):

            # Step 2: Remove Unnecessary Replacement Words 
            perturbed_text = self.remove_unnecessary_words(perturbed_text, attacked_text) 
            print (f'SSPAttack Perturbed Text with unnecessarily perturbed words reverted back to original. \n {perturbed_text}') 
            self.ceattack_logger.debug(f'SSPAttack Perturbed Text with unnecessarily perturbed words reverted back to original. \n {perturbed_text}')

            
            

            # Step 3: Push Substitution Words towards Original Words
            perturbed_text = self.push_words_towards_original(perturbed_text, attacked_text)
            print (f'SSPAttack Perturbed Text with perturbed words modified to more semantically similar ones. \n {perturbed_text}') 
            self.ceattack_logger.debug(f'SSPAttack Perturbed Text with perturbed words modified to more semantically similar ones. \n {perturbed_text}')

            
            results, search_over = self.get_goal_results([perturbed_text])
            
            self.number_of_queries+= 1 # only 1 sample at the time
            
            final_result = results[0]

             

            if final_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                
                sim_final_original, sim_final_pert = self.use_constraint.encode([attacked_text.text, final_result.attacked_text.text])

                if not isinstance(sim_final_original, torch.Tensor):
                    sim_final_original = torch.tensor(sim_final_original)

                if not isinstance(sim_final_pert, torch.Tensor):
                    sim_final_pert = torch.tensor(sim_final_pert)

                sim_score = self.sentence_encoder_use.sim_metric(sim_final_original.unsqueeze(0), sim_final_pert.unsqueeze(0)).item()
                
                
                if sim_score <  (1 - (self.similarity_threshold) / math.pi):
                    continue

                
                
                print (f'SSPAttack Perturbed Text final result. \n {final_result.attacked_text}') 
                self.ceattack_logger.debug(f'SSPAttack Perturbed Text final result. \n {final_result.attacked_text}')



                return final_result
            
        return initial_result

    def random_initialization(self, text):
        words = text.words
        tmp_text = text
        size_text = len(text.words)
        start_i = 0
        while start_i < size_text:
            
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

 

    def remove_unnecessary_words(self, perturbed_text, original_text, check_skip=False):
        
        candidate_set = []
        word_importance_scores = [] 
        
        for i, (perturbed_word, original_word) in enumerate(zip(perturbed_text.words, original_text.words)):
            if perturbed_word != original_word:
                
                temp_text = perturbed_text.replace_word_at_index(i, original_word)
                
                
                current_goal_status = self.check_model_status(temp_text,check_skip)
                self.number_of_queries+=1 
                
                if current_goal_status == GoalFunctionResultStatus.SUCCEEDED:
                    candidate_set.append((i, temp_text))
                    sim_remove_unnecessary_org, sim_remove_unnecessary_pert = self.sentence_encoder_use.encode([original_text.text, temp_text.text])

                    if not isinstance(sim_remove_unnecessary_org, torch.Tensor):
                        sim_remove_unnecessary_org = torch.tensor(sim_remove_unnecessary_org)

                    if not isinstance(sim_remove_unnecessary_pert, torch.Tensor):
                        sim_remove_unnecessary_pert = torch.tensor(sim_remove_unnecessary_pert)

                    sim_score = self.sentence_encoder_use.sim_metric(sim_remove_unnecessary_org.unsqueeze(0), sim_remove_unnecessary_pert.unsqueeze(0))
                    word_importance_scores.append((i, sim_score))


        word_importance_scores.sort(key=lambda x: x[1], reverse=True)
        
        for idx, _ in word_importance_scores:
            temp_text2 = perturbed_text.replace_word_at_index(idx, original_text.words[idx])
            temp_text2.attack_attrs['modified_indices'].remove(idx)
            
            current_goal_status = self.check_model_status(temp_text2,check_skip)
            self.number_of_queries+=1
            
 
            if current_goal_status == GoalFunctionResultStatus.SUCCEEDED:
                
                perturbed_text = temp_text2
                
            else:
                break
            
        return perturbed_text

    def get_vector(self, embedding, word):
        if isinstance(word, str):
            if word in embedding._word2index:
                word_index = embedding._word2index[word]
            else:
                return None  
            
        else:
            word_index = word

        vector = embedding.embedding_matrix[word_index]
        return torch.tensor(vector).to(device)
    

    def push_words_towards_original(self, perturbed_text, original_text, check_skip=False):
        
        distances = []
        for i, (perturbed_word, original_word) in enumerate(zip(perturbed_text.words, original_text.words)):
            if perturbed_word != original_word:
                
                perturbed_vec = self.get_vector(self.embedding, perturbed_word)
                if perturbed_vec is None:
                    continue  
                
                original_vec = self.get_vector(self.embedding, original_word)
                if original_vec is None:
                    continue  
                
                distance = np.linalg.norm(perturbed_vec.cpu().numpy() - original_vec.cpu().numpy())
                distances.append((i, distance))

        if not distances:
            return perturbed_text


        distances.sort(key=lambda x: x[1])
        indices, dist_values = zip(*distances)
        exp_dist_values = np.exp(dist_values)
        probabilities = exp_dist_values / np.sum(exp_dist_values)
        
        while len(indices) > 0:
            i = np.random.choice(indices, p=probabilities)
             
            perturbed_word = perturbed_text.words[i]
            original_word = original_text.words[i]

            sentence_replaced = self.get_transformations(original_text, original_text=original_text, indices_to_modify=[i])
            synonyms = [s.words[i] for s in sentence_replaced]


             
            k = 10   
            top_k_synonyms_indexes  = self.embedding.nearest_neighbours(self.embedding._word2index[original_word], topn=k)
            top_k_synonyms = [self.embedding._index2word[index] for index in top_k_synonyms_indexes]

             
            max_similarity = -float('inf')
            w_bar = None
            temp_text_bar = None
            filtered_synonyms = None 
            for synonym in top_k_synonyms:
                if perturbed_word == synonym:
                    continue   
                temp_text2 = perturbed_text.replace_word_at_index(i, synonym)
 
                current_goal_status = self.check_model_status(temp_text2,check_skip)
                self.number_of_queries+=1 
 
                if current_goal_status == GoalFunctionResultStatus.SUCCEEDED:
                     
                    sim_remove_unnecessary_org, sim_remove_unnecessary_pert = self.sentence_encoder_use.encode([original_word, synonym])

                    if not isinstance(sim_remove_unnecessary_org, torch.Tensor):
                        sim_remove_unnecessary_org = torch.tensor(sim_remove_unnecessary_org)

                    if not isinstance(sim_remove_unnecessary_pert, torch.Tensor):
                        sim_remove_unnecessary_pert = torch.tensor(sim_remove_unnecessary_pert)

                    sim_score = self.sentence_encoder_use.sim_metric(sim_remove_unnecessary_org.unsqueeze(0), sim_remove_unnecessary_pert.unsqueeze(0)).item()
                     
                    if sim_score > max_similarity:
                        max_similarity = sim_score
                        w_bar = synonym
                        temp_text_bar = temp_text2

            
            if w_bar: 
                number_entries = len(self.embedding.nn_matrix[self.embedding._word2index[original_word]] )
                 
                all_synonyms = self.embedding.nearest_neighbours(self.embedding._word2index[original_word], topn=number_entries)
                all_synonyms = [self.embedding._index2word[index] for index in all_synonyms]
            
                 
                filtered_synonyms = []
                for synonym in all_synonyms:
                    if perturbed_word == synonym or w_bar == synonym  :
                        continue  
                    sim_w_bar, sim_synonym = self.sentence_encoder_use.encode([w_bar, synonym])
                    sim_org, sim_synonym_org = self.sentence_encoder_use.encode([original_word, synonym])

                    if not isinstance(sim_w_bar, torch.Tensor):
                        sim_w_bar = torch.tensor(sim_w_bar)
                    if not isinstance(sim_synonym, torch.Tensor):
                        sim_synonym = torch.tensor(sim_synonym)
                    if not isinstance(sim_org, torch.Tensor):
                        sim_org = torch.tensor(sim_org)
                    if not isinstance(sim_synonym_org, torch.Tensor):
                        sim_synonym_org = torch.tensor(sim_synonym_org)

                    sim_score_w_bar = self.sentence_encoder_use.sim_metric(sim_w_bar.unsqueeze(0), sim_synonym.unsqueeze(0)).item()
                    sim_score_org = self.sentence_encoder_use.sim_metric(sim_org.unsqueeze(0), sim_synonym_org.unsqueeze(0)).item()

                    if sim_score_w_bar > sim_score_org:
                        filtered_synonyms.append((sim_score_w_bar, synonym))

            if  filtered_synonyms: 
                filtered_synonyms.sort(key=lambda item: item[0], reverse=True) 
                
                for _, synonym in filtered_synonyms:
                    temp_text2 = perturbed_text.replace_word_at_index(i, synonym) 
                    

                    current_goal_status = self.check_model_status(temp_text2,check_skip)
                    self.number_of_queries+=1
                    
                    if current_goal_status == GoalFunctionResultStatus.SUCCEEDED:
                        perturbed_text = temp_text2
                        break
                    
            


            idx = indices.index(i)
            indices = indices[:idx] + indices[idx + 1:] 
            
            
            probabilities = np.delete(probabilities, idx)
            probabilities /= np.sum(probabilities)   
            
        return perturbed_text 

    def get_transformations(self, text, index):
        return self.transformation(text, index)

    def get_similarity(self, word1, word2):
        return self.transformation.get_cosine_similarity(word1, word2)
    
    @property
    def is_black_box(self):
        return True
