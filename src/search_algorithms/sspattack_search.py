
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
        self.number_of_queries = 0 # raw counter mostly for infucntion debugging purposes
        attacked_text = initial_result.attacked_text

        print ('goal_function',self.goal_function)
        
        # Step 1: Initialization
        number_samples = self.num_transformations
        # self.number_of_queries+=number_samples + 1 # checking the original sample if it's correct, then num samples perturbations to find adv
        # self.goal_function.num_queries +=number_samples + 1
        # self.goal_function.num_queries += self.num_transformations
        # print ('self.goal_function.num_queries 1',self.goal_function.num_queries)
        # print ('self.goal_function.num_queries',a,'self.number_of_queries',self.number_of_queries)
        perturbed_text = [self.random_initialization(attacked_text) for i in range(number_samples)]
        
        results, search_over = self.get_goal_results(perturbed_text) # automatically keeps track of queries
        self.number_of_queries+=number_samples + 1
        # self.goal_function.num_queries += self.num_transformations
        print ('self.goal_function.num_queries 1',self.goal_function.num_queries)
        
        
        results_success = [result for result in results if result.ground_truth_output!=result.output] 

        results_success=[]
        perturbed_text_success = []
        for i,result in enumerate(results):
            if result.ground_truth_output!=result.output:
                results_success.append(result)
                perturbed_text_success.append(perturbed_text[i])


        print ('returnign failed?')
        if len(results_success) == 0:
            final_result = results[0]
            # final_result.num_queries = self.number_of_queries
            # self.goal_function.num_queries = self.number_of_queries 
            print ('self.goal_function.num_queries 2',self.goal_function.num_queries,'self.number_of_queries ',self.number_of_queries )
            
            # print ('self.goal_function.num_queries2',self.goal_function.num_queries,'self.number_of_queries',self.number_of_queries)
            return final_result # return a random result that wasnt perturbed to show it failed.

        perturbed_text = perturbed_text_success[0]
        results = results_success[0]
 

        # check that sample is adversarial

        print ('attacked_text',attacked_text)
        print ('perturbed_text',perturbed_text)

                 

        
        
        # Main iteration loop
        for _ in range(self.max_iterations):
            # Step 2: Remove Unnecessary Replacement Words
            
            perturbed_text = self.remove_unnecessary_words(perturbed_text, attacked_text)
            print ('perturned+text',perturbed_text)

            # if attacked_text.words == perturbed_text.words:
            #     print ('should we skipp?')
            #     sys.exit() 

            # Step 3: Push Substitution Words towards Original Words
            perturbed_text = self.push_words_towards_original(perturbed_text, attacked_text)
            print ('perturned+text2',perturbed_text) 
            # if attacked_text == perturbed_text:
            #     print ('should we skipp 2?')
            #     sys.exit() 
            # Check if attack is successful
            results, search_over = self.get_goal_results([perturbed_text])
            # perturbed_result = initial_result.goal_function.call_model([perturbed_text])[0]
            # print ('results',results)
            self.number_of_queries+= 1 # only 1 sample at the time
            print ('self.goal_function.num_queries preend',self.goal_function.num_queries,'self.number_of_queries ',self.number_of_queries )
            
            # add semantic sim filter

            # this checks the generated test against the actual final use constraint
            
              

            final_result = results[0]

             

            if final_result.goal_status == GoalFunctionResultStatus.SUCCEEDED:
                print ('attacked_text.text, final_result.attacked_text.text')
                print ('attk text',attacked_text.text)
                print ('final sre',final_result.attacked_text.text)
                sim_final_original, sim_final_pert = self.use_constraint.encode([attacked_text.text, final_result.attacked_text.text])

                if not isinstance(sim_final_original, torch.Tensor):
                    sim_final_original = torch.tensor(sim_final_original)

                if not isinstance(sim_final_pert, torch.Tensor):
                    sim_final_pert = torch.tensor(sim_final_pert)

                sim_score = self.sentence_encoder_use.sim_metric(sim_final_original.unsqueeze(0), sim_final_pert.unsqueeze(0)).item()
                print ('sim_score',sim_score, (1 - (self.similarity_threshold) / math.pi))
                if sim_score <  (1 - (self.similarity_threshold) / math.pi):
                    continue

                # final_result.num_queries = self.number_of_queries
                # self.goal_function.num_queries = self.number_of_queries
                print ('final_result.num_queries',final_result.num_queries)
                print ('final,self.number_of_queries',self.number_of_queries)
                print ('final,self.goal_function.num_queries ',self.goal_function.num_queries)
                print ('self.goal_function.num_queries endend',self.goal_function.num_queries,'self.number_of_queries ',self.number_of_queries )
            
                
                # sys.exit()
                print ('final_result',final_result.attacked_text)
                print ('final_result.attacked_text.attack_attrs',final_result.attacked_text.attack_attrs)
                print ('final_result.attacked_text.attack_attrs[original_index_map]',final_result.attacked_text.attack_attrs['original_index_map'])
                # print ('final_result',final_result.original_text)
                # print ('final_result',final_result.perturbed_text)
                # print ('final_result.perturbed_result.attack_attrs',final_result.perturbed_result.attack_attrs)
                # if len(final_result.attacked_text.attack_attrs['newly_modified_indices']) == 0:
                #     final_result.attacked_text.attack_attrs['newly_modified_indices'] = {0}
                # if len(final_result.attacked_text.attack_attrs['modified_indices']) == 0:
                #     # final_result.attacked_text.attack_attrs['modified_indices'] = {0}
                #     return initial_result

                return final_result
        print ('just aviod everything')
        return initial_result

    def random_initialization(self, text):
        words = text.words
        tmp_text = text
        size_text = len(text.words)
        start_i = 0
        while start_i < size_text:
            # print ('start tmp text',tmp_text)
            words = tmp_text.words
            pos_tags = nltk.pos_tag(words)   
            # print ('pos_tags',pos_tags)
            if pos_tags[start_i][1].startswith(('VB', 'NN', 'JJ', 'RB')): 
                # print ('pos_tags[start_i][1]',pos_tags[start_i][1])
                replaced_with_synonyms = self.get_transformations(tmp_text, original_text=tmp_text,indices_to_modify=[start_i])
                # print ('replaced_with_synonyms',replaced_with_synonyms)
                if replaced_with_synonyms:
                    tmp_text = random.choice(replaced_with_synonyms)
                else:
                    pass
                
            start_i+=1
            
        adv_text = tmp_text
        return adv_text

 

    def remove_unnecessary_words(self, perturbed_text, original_text, check_skip=False):
        # Step 1: Identify words to replace back
        candidate_set = []
        word_importance_scores = [] 
        # print ('original_text',original_text)
        # print ('perturbed_text',perturbed_text)
        for i, (perturbed_word, original_word) in enumerate(zip(perturbed_text.words, original_text.words)):
            if perturbed_word != original_word:
                # Replace perturbed_word with original_word
                temp_text = perturbed_text.replace_word_at_index(i, original_word)

                # Step 2: Check if still adversarial and calculate semantic similarity
                # model_outputs = self.goal_function._call_model([temp_text])
                # current_goal_status = self.goal_function._get_goal_status(
                #     model_outputs[0], temp_text, check_skip=check_skip
                # ) # this does keep track of queries so i have to keep track myself
                
                current_goal_status = self.check_model_status(temp_text,check_skip)
                self.number_of_queries+=1 
                print ('self.goal_function.num_queries 2',self.goal_function.num_queries,'self.number_of_queries ',self.number_of_queries )
            
                # print ('temp_text',temp_text,i,current_goal_status,GoalFunctionResultStatus.SUCCEEDED)
                if current_goal_status == GoalFunctionResultStatus.SUCCEEDED:
                    candidate_set.append((i, temp_text))
                    sim_remove_unnecessary_org, sim_remove_unnecessary_pert = self.sentence_encoder_use.encode([original_text.text, temp_text.text])

                    if not isinstance(sim_remove_unnecessary_org, torch.Tensor):
                        sim_remove_unnecessary_org = torch.tensor(sim_remove_unnecessary_org)

                    if not isinstance(sim_remove_unnecessary_pert, torch.Tensor):
                        sim_remove_unnecessary_pert = torch.tensor(sim_remove_unnecessary_pert)

                    sim_score = self.sentence_encoder_use.sim_metric(sim_remove_unnecessary_org.unsqueeze(0), sim_remove_unnecessary_pert.unsqueeze(0))
                    word_importance_scores.append((i, sim_score))

        # Step 3: Sort word importance scores in descending order and restore original words
        word_importance_scores.sort(key=lambda x: x[1], reverse=True)
        print ('attack_attrs ',perturbed_text.attack_attrs,perturbed_text  ) 
        print ('replace indexs',word_importance_scores)
        for idx, _ in word_importance_scores:
            temp_text2 = perturbed_text.replace_word_at_index(idx, original_text.words[idx])
            temp_text2.attack_attrs['modified_indices'].remove(idx)
            print ('temp_text2_word_imp',idx,temp_text2.attack_attrs,temp_text2)
            # print ('original_index_map',temp_text2.attack_attrs.original_index_map)
            
            # model_outputs = self.goal_function._call_model([temp_text2])
            # current_goal_status = self.goal_function._get_goal_status(
            #     model_outputs[0], temp_text2, check_skip=check_skip
            # ) # doest keep track of queries
            # self.goal_function.num_queries+=1
            current_goal_status = self.check_model_status(temp_text2,check_skip)
            self.number_of_queries+=1 
            print ('self.goal_function.num_queries 3',self.goal_function.num_queries,'self.number_of_queries ',self.number_of_queries )
            
            # print ('temp_text2',temp_text2,current_goal_status, GoalFunctionResultStatus.SUCCEEDED)

 
            if current_goal_status == GoalFunctionResultStatus.SUCCEEDED:
                # If perturbed_text is no longer adversarial, revert the last change
                perturbed_text = temp_text2
                # perturbed_text = perturbed_text.replace_word_at_index(idx, perturbed_text.words[idx])
            else:
                break
        # print ('original_text',original_text)
        # print ('perturbed_text',perturbed_text) 
        return perturbed_text

    def get_vector(self, embedding, word):
        if isinstance(word, str):
            if word in embedding._word2index:
                word_index = embedding._word2index[word]
            else:
                return None  # Word not found in the dictionary
        else:
            word_index = word

        vector = embedding.embedding_matrix[word_index]
        return torch.tensor(vector).to(device)
        # return torch.tensor(vector).to(textattack.shared.utils.device)

    def push_words_towards_original(self, perturbed_text, original_text, check_skip=False):
        # Step 1: Calculate Euclidean distances and sampling probabilities
        distances = []
        for i, (perturbed_word, original_word) in enumerate(zip(perturbed_text.words, original_text.words)):
            if perturbed_word != original_word:
                # Using the get_vector function
                perturbed_vec = self.get_vector(self.embedding, perturbed_word)
                if perturbed_vec is None:
                    continue  # Skip to the next word
                original_vec = self.get_vector(self.embedding, original_word)
                if original_vec is None:
                    continue  # Skip to the next word
                distance = np.linalg.norm(perturbed_vec.cpu().numpy() - original_vec.cpu().numpy())
                distances.append((i, distance))

        if not distances:
            return perturbed_text

        # Normalize distances to get probabilities
        distances.sort(key=lambda x: x[1])
        indices, dist_values = zip(*distances)
        exp_dist_values = np.exp(dist_values)
        probabilities = exp_dist_values / np.sum(exp_dist_values)
        print ('probabilities',probabilities)

        # temp_perturbed_text = copy.deepcopy(perturbed_text)
        
        # Step 2: Iterate with sampling based on the probabilities 
        while len(indices) > 0:
            i = np.random.choice(indices, p=probabilities)
            print ('indices',indices,i)
            perturbed_word = perturbed_text.words[i]
            original_word = original_text.words[i]

            sentence_replaced = self.get_transformations(original_text, original_text=original_text, indices_to_modify=[i])
            synonyms = [s.words[i] for s in sentence_replaced]


            # Get top k synonyms
            k = 10  # Number of synonyms to sample
            top_k_synonyms_indexes  = self.embedding.nearest_neighbours(self.embedding._word2index[original_word], topn=k)
            top_k_synonyms = [self.embedding._index2word[index] for index in top_k_synonyms_indexes]

            # Find the best anchor synonym with the highest semantic similarity
            max_similarity = -float('inf')
            w_bar = None
            temp_text_bar = None
            filtered_synonyms = None
            print ('top_k_synonyms',top_k_synonyms)
            # temp_text2 = copy.deepcopy(perturbed_text)
            for synonym in top_k_synonyms:
                if perturbed_word == synonym:
                    continue # skip swapping the same word
                print ('synonym',i,synonym)
                # temp_text2 = copy.deepcopy(perturbed_text)
                temp_text2 = perturbed_text.replace_word_at_index(i, synonym)

                # Check if the substitution still results in an adversarial example
                # model_outputs = self.goal_function._call_model([temp_text2])
                # current_goal_status = self.goal_function._get_goal_status(
                #     model_outputs[0], temp_text2, check_skip=check_skip
                # ) # doesnt keep track of queries
                # self.goal_function.num_queries+=1
                current_goal_status = self.check_model_status(temp_text2,check_skip)
                self.number_of_queries+=1 
                print ('self.goal_function.num_queries 4',self.goal_function.num_queries,'self.number_of_queries ',self.number_of_queries )
            
                print ('temp_text2_top_k_syn',i,synonym,temp_text2.attack_attrs,temp_text2,current_goal_status , GoalFunctionResultStatus.SUCCEEDED)

                if current_goal_status == GoalFunctionResultStatus.SUCCEEDED:
                    # Compute semantic similarity at the word level
                    sim_remove_unnecessary_org, sim_remove_unnecessary_pert = self.sentence_encoder_use.encode([original_word, synonym])

                    if not isinstance(sim_remove_unnecessary_org, torch.Tensor):
                        sim_remove_unnecessary_org = torch.tensor(sim_remove_unnecessary_org)

                    if not isinstance(sim_remove_unnecessary_pert, torch.Tensor):
                        sim_remove_unnecessary_pert = torch.tensor(sim_remove_unnecessary_pert)

                    sim_score = self.sentence_encoder_use.sim_metric(sim_remove_unnecessary_org.unsqueeze(0), sim_remove_unnecessary_pert.unsqueeze(0)).item()
                    print ('sim scores push towards orgin')
                    if sim_score > max_similarity:
                        max_similarity = sim_score
                        w_bar = synonym
                        temp_text_bar = temp_text2

            
            if w_bar:# is None:
                  # Skip this index if no suitable anchor synonym is found
            
                number_entries = len(self.embedding.nn_matrix[self.embedding._word2index[original_word]] )
                print ('num entries',number_entries)
                all_synonyms = self.embedding.nearest_neighbours(self.embedding._word2index[original_word], topn=number_entries)
                all_synonyms = [self.embedding._index2word[index] for index in all_synonyms]
            
                print ('all_synonyms',all_synonyms)
                filtered_synonyms = []
                for synonym in all_synonyms:
                    if perturbed_word == synonym or w_bar == synonym  :
                        continue # skip swapping/checking the same word and the anchor word
                    # Compute semantic similarity with w_bar and original_word
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
                # continue  # Skip this index if no suitable synonym is found

                # Sort the filtered synonyms by their semantic similarity score in descending order
                filtered_synonyms.sort(key=lambda item: item[0], reverse=True)
                print ('filtered_synonyms',filtered_synonyms)
                
                

                print ('perturbed text',perturbed_text.attack_attrs,perturbed_text)
                for _, synonym in filtered_synonyms:
                    temp_text2 = perturbed_text.replace_word_at_index(i, synonym) 
                    # temp_text2.attack_attrs['modified_indices'].remove(i)
                    print ('temp_text2_filtered_syn',i,temp_text2.attack_attrs,temp_text2)
                    # Check if the substitution still results in an adversarial example
                    # model_outputs = self.goal_function._call_model([temp_text2]) 
                    # current_goal_status = self.goal_function._get_goal_status(
                    #     model_outputs[0], temp_text2, check_skip=check_skip
                    # ) 
                    # self.goal_function.num_queries+=1
                    current_goal_status = self.check_model_status(temp_text2,check_skip)
                    self.number_of_queries+=1

                    print ('self.goal_function.num_queries 5',self.goal_function.num_queries,'self.number_of_queries ',self.number_of_queries )
            
                    print ('temp_text2',temp_text2,current_goal_status, GoalFunctionResultStatus.SUCCEEDED)
                    if current_goal_status == GoalFunctionResultStatus.SUCCEEDED:
                        perturbed_text = temp_text2
                        break
                    
            
            print ('perturbed_text',perturbed_text)  
            idx = indices.index(i)
            indices = indices[:idx] + indices[idx + 1:] 
            print ('indices2',indices,idx)
            
            probabilities = np.delete(probabilities, idx)
            probabilities /= np.sum(probabilities)   
        # sys.exit()
        return perturbed_text 

    def get_transformations(self, text, index):
        return self.transformation(text, index)

    def get_similarity(self, word1, word2):
        return self.transformation.get_cosine_similarity(word1, word2)
    
    @property
    def is_black_box(self):
        return True
