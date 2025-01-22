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
        else:
            # Download the file
            print(f"Downloading file to: {output_path}")
            gdown.download(url, output_path, quiet=False)
            print("Download complete.")
        
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
        print ('tagset',tagset)
        _, pos_list = zip(*word_n_pos_list)
        return pos_list

    def perform_search(self, initial_result):
        self.number_of_queries = 0
        attacked_text = initial_result.attacked_text

        print ('goal_function',self.goal_function)
        
        

        # get_vector(self, self.embedding, word): # word can either be a str or the index eqivalant of embed_content[word_idx_dict[word] ]

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
        # if len_text < sim_score_window:
        #     sim_score_threshold = 0.1
        # half_sim_score_window = (sim_score_window - 1) // 2
        num_queries = 1
        rank = {}
        words_perturb = []
        # pos_ls = criteria.get_pos(text_ls)
        # print ('pos_ls',pos_ls)
        # pos_pref = ["ADJ", "ADV", "VERB", "NOUN"]
        # for pos in pos_pref:
        #     for i in range(len(pos_ls)):
        #         if pos_ls[i] == pos and len(text_ls[i]) > 2:
        #             words_perturb.append((i, text_ls[i]))
        # print ('words_perturb',words_perturb)
        # pos_ls = criteria.get_pos(text_ls)
        pos_tags = nltk.pos_tag(text_ls)   
        print ('pos_tags',pos_tags)
        for i, (word, pos_tag) in enumerate(pos_tags):
            if pos_tag.startswith(('VB', 'NN', 'JJ', 'RB')) and len(word) > 2:
                words_perturb.append((i, word))

        random.shuffle(words_perturb)
        words_perturb = words_perturb[:top_k_words]



        print ('words_perturb',words_perturb) 
        words_perturb_indices = [idx for idx, word in words_perturb]



        # Step 1: Initialization
        number_samples = self.num_transformations # 2
        self.number_of_queries+=number_samples + 1 # checking the original sample if it's correct, then num samples perturbations to find adv
        self.goal_function.num_queries = self.number_of_queries
        # self.goal_function.num_queries += number_samples + 1
        perturbed_text = [self.random_initialization(attacked_text,words_perturb_indices) for i in range(number_samples)]
        print ('perturbed_text',perturbed_text)
        results, search_over = self.get_goal_results(perturbed_text)
        
         
        
        # results_success = [result for result in results if result.ground_truth_output!=result.output] 

        results_success=[]
        perturbed_text_success = []
        for i,result in enumerate(results):
            if result.ground_truth_output!=result.output:
                results_success.append(result)
                perturbed_text_success.append(perturbed_text[i])


        print ('returnign failed?')
        if len(results_success) == 0:
            flag = 0
        else:
            flag = 1

            #     final_result = results[0]
                
            #     self.goal_function.num_queries = self.number_of_queries
            #     self.goal_function.model.reset_inference_steps()
            #     return final_result # return a random result that wasnt perturbed to show it failed.

            perturbed_text = perturbed_text_success[0]
            # results = results_success[0]
    

            # check that sample is adversarial

            print ('attacked_text',attacked_text)
            print ('perturbed_text',perturbed_text)

            random_text = perturbed_text.words



        words_perturb_idx= []
        words_perturb_embed = []
        words_perturb_doc_idx = []
        for idx, word in words_perturb:
            if word in word_idx_dict:
                words_perturb_doc_idx.append(idx)
                words_perturb_idx.append(word2idx[word])
                # print ('[float(num) for num in embed_content[ word_idx_dict[word] ]', embed_content[ word_idx_dict[word]] )
                words_perturb_embed.append([float(num) for num in embed_content[ word_idx_dict[word] ]])
                # print ('words_perturb_embed',words_perturb_embed) 

        words_perturb_embed_matrix = np.asarray(words_perturb_embed)


        synonym_words,synonym_values=[],[]
        for idx in words_perturb_idx:
            res = list(zip(*(cos_sim[idx])))
            
            # print ('res',res, len(res),len(res[0]),len(res[1]))
            res[0] = res[0][:self.n_embeddings]
            res[1] = res[1][:self.n_embeddings]
            # [:self.n_embeddings]
            # print ('res',len(res[0]),len(res[1]))
            # sys.exit()
            temp=[]
            for ii in res[1]:
                temp.append(idx2word[ii])
            print ('temp syn words',idx2word[idx],temp, len(temp))
            synonym_words.append(temp)
            temp=[]
            for ii in res[0]:
                temp.append(ii)
            print ('temp syn values',idx2word[idx],temp, len(temp))
            synonym_values.append(temp)

        synonyms_all = []
        synonyms_dict = defaultdict(list)
        for idx, word in words_perturb:
            if word in word2idx:
                synonyms = synonym_words.pop(0)
                if synonyms:
                    synonyms_all.append((idx, synonyms))
                    synonyms_dict[word] = synonyms

        # print ('synonym_words',synonym_words)
        print ('synonyms_all',synonyms_all)
        ###################### erlier code ##########################
        # qrs = 1 # qrs start at 1 because we already had to use 1 query to detect if sample gets classified correctly
        # num_changed = 0
        # flag = 0
        # th = 0
        # while qrs < len(text_ls):
        #     print ('qrs1',qrs)
        #     random_text = text_ls[:]
        #     for i in range(len(synonyms_all)):
        #         idx = synonyms_all[i][0]
        #         syn = synonyms_all[i][1]
        #         random_text[idx] = random.choice(syn)
        #         if i >= th:
        #             break
        #     print ('random_text 1',random_text)
        #     print ('attacked_text 1',attacked_text)
        #     print ('text_ls 1',text_ls)
        #     # random_text_joint = attacked_text.generate_new_attacked_text(random_text) 
        #     # model_outputs = self.goal_function._call_model([random_text_joint])
        #     # current_goal_status = self.goal_function._get_goal_status(
        #     #     model_outputs[0], random_text_joint, check_skip=False
        #     # )
        #     # self.number_of_queries+=1
            
        #     random_text_joint = attacked_text.generate_new_attacked_text(random_text)
        #     results_adv_initial, search_over = self.get_goal_results([random_text_joint])
        #     # pr = get_attack_result([random_text], predictor, orig_label, batch_size)
        #     if search_over: 
        #         self.goal_function.model.reset_inference_steps()
        #         return initial_result

        #     qrs+=1
        #     self.number_of_queries+=1
        #     self.goal_function.num_queries = self.number_of_queries
        #     th +=1
        #     if th > len_text:
        #         break
        #     # if np.sum(pr)>0:
        #     # if current_goal_status == GoalFunctionResultStatus.SUCCEEDED:
        #     # print ('results first',results)
        #     # if len(results) == 0: 
        #     #     print ('returning initial result 1', initial_result)
        #     #     return initial_result

        #     if results_adv_initial[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
        #         flag = 1
        #         break
        # old_qrs = qrs
        # print ('old_qrs',old_qrs)
        ######################################################
        qrs = number_samples + 1
        old_qrs = qrs  
        print ('old_qrs',old_qrs)

        # while qrs < old_qrs + 2500 and flag == 0:

        # we remove this part because althou if tries all possible queries to find a solution, to make sure we stay in buget we 
        # focus only on examples that we found an adversarial example quickly then optimize it's semantic similarity.
        # while qrs < budget and flag == 0:
        #     print ('qrs2',qrs)
        #     random_text = text_ls[:]
        #     for j in range(len(synonyms_all)):
        #         idx = synonyms_all[j][0]
        #         syn = synonyms_all[j][1]
        #         random_text[idx] = random.choice(syn)
        #         if j >= len_text:
        #             break
        #     # pr = get_attack_result([random_text], predictor, orig_label, batch_size) 
        #     print ('random_text 2',random_text)
        #     print ('attacked_text 2',attacked_text)
        #     print ('text_ls 2',text_ls)
        #     # random_text_joint = attacked_text.generate_new_attacked_text(random_text) 
        #     # model_outputs = self.goal_function._call_model([random_text_joint])
        #     # current_goal_status = self.goal_function._get_goal_status(
        #     #     model_outputs[0], random_text_joint, check_skip=False
        #     # )
        #     # self.number_of_queries+=1

        #     random_text_joint = attacked_text.generate_new_attacked_text(random_text)
        #     results_adv_initial, search_over = self.get_goal_results([random_text_joint])
        #     if search_over: 
        #         self.goal_function.model.reset_inference_steps()
        #         return initial_result
        #     qrs+=1
        #     self.number_of_queries+=1
        #     self.goal_function.num_queries = self.number_of_queries
        #     # if np.sum(pr)>0:
        #     # if current_goal_status == GoalFunctionResultStatus.SUCCEEDED:
        #     print ('results_adv_initial second',results_adv_initial)


        #     # print ('returning failed result because flag==0')
        #     # results, search_over = self.get_goal_results([attacked_text])
        #     # return results[0]

        #     # if len(results) == 0: 
        #     #     print ('returning initial result 2', initial_result)
        #     #     return initial_result

        #     if results_adv_initial[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
        #         flag = 1
        #         break
            
            
        print ('flag',flag) 
        if flag == 1:
            changed = 0
            for i in range(len(text_ls)):
                if text_ls[i]!=random_text[i]:
                    changed+=1
             

            print ('original_random_text',random_text)
            while True:
                choices = []

                for i in range(len(text_ls)):
                    if random_text[i] != text_ls[i]:
                        new_text = random_text[:]
                        new_text[i] = text_ls[i]
                        # print ('text_ls, [new_text], -1, sim_score_window, sim_predictor',text_ls, [new_text], -1, sim_score_window, sim_predictor)
                        
                        


                        new_text_joint = attacked_text.generate_new_attacked_text(new_text) 


                        print ('attacked_text',attacked_text)
                        print ('new_text_joint',new_text_joint)
                        if attacked_text.text == new_text_joint.text: # is word sub leads to perturbation being same as original sample skip
                            continue
                        

                        sim_remove_unnecessary_org, sim_remove_unnecessary_pert = self.sentence_encoder_use.encode([attacked_text.text, new_text_joint.text])

                        if not isinstance(sim_remove_unnecessary_org, torch.Tensor):
                            sim_remove_unnecessary_org = torch.tensor(sim_remove_unnecessary_org)

                        if not isinstance(sim_remove_unnecessary_pert, torch.Tensor):
                            sim_remove_unnecessary_pert = torch.tensor(sim_remove_unnecessary_pert)

                        semantic_sims = self.sentence_encoder_use.sim_metric(sim_remove_unnecessary_org.unsqueeze(0), sim_remove_unnecessary_pert.unsqueeze(0))
                        
                        print ('semantic_sims1',semantic_sims)
                        # semantic_sims = calc_sim(text_ls, [new_text], -1, sim_score_window, sim_predictor)
                        # model_outputs = self.goal_function._call_model([new_text_joint])
                        # current_goal_status = self.goal_function._get_goal_status(
                        #     model_outputs[0], new_text_joint, check_skip=False
                        # )
                        # self.number_of_queries+=1
                        # random_text_joint = attacked_text.generate_new_attacked_text(new_text_joint)
                        results, search_over = self.get_goal_results([new_text_joint])
                        if search_over: 
                            self.goal_function.model.reset_inference_steps()
                            return initial_result
                        print ('qrs3',qrs)
                        qrs+=1
                        self.number_of_queries+=1
                        self.goal_function.num_queries = self.number_of_queries
                        # self.goal_function.num_queries+=1
                        # if current_goal_status == GoalFunctionResultStatus.SUCCEEDED:
                        if results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                            choices.append((i,semantic_sims[0]))
 
                        # qrs+=1
                        # pr = get_attack_result([new_text], predictor, orig_label, batch_size)
                        # if np.sum(pr) > 0:
                        #     choices.append((i,semantic_sims[0]))

                print ('choices', choices) 
                if len(choices) > 0:
                    choices.sort(key = lambda x: x[1])
                    choices.reverse()
                    for i in range(len(choices)):
                        new_text = random_text[:]
                        new_text[choices[i][0]] = text_ls[choices[i][0]]

                        # new_text_joint = attacked_text.generate_new_attacked_text(new_text) 
                        # model_outputs = self.goal_function._call_model([new_text_joint])
                        # current_goal_status = self.goal_function._get_goal_status(
                        #     model_outputs[0], new_text_joint, check_skip=False
                        # )
                        # self.number_of_queries+=1

                        new_text_joint = attacked_text.generate_new_attacked_text(new_text)
                        if attacked_text.text == new_text_joint.text: # is word sub leads to perturbation being same as original sample skip
                            continue

                        results, search_over = self.get_goal_results([new_text_joint])
                        if search_over: 
                            self.goal_function.model.reset_inference_steps()
                            return initial_result

                        qrs+=1
                        self.number_of_queries+=1
                        self.goal_function.num_queries = self.number_of_queries
                        # self.goal_function.num_queries+=1
                        # if current_goal_status != GoalFunctionResultStatus.SUCCEEDED:
                        if results[0].goal_status != GoalFunctionResultStatus.SUCCEEDED:
                            break
                        random_text[choices[i][0]] = text_ls[choices[i][0]]

                        # pr = get_attack_result([new_text], predictor, orig_label, batch_size)
                        # qrs+=1
                        # if pr[0] == 0:
                        #     break
                        # random_text[choices[i][0]] = text_ls[choices[i][0]]

                if len(choices) == 0:
                    break
            
            print ('after choices random_text',random_text) 
            changed_indices = [] 
            num_changed = 0
            for i in range(len(text_ls)):
                if text_ls[i]!=random_text[i]:
                    changed_indices.append(i)
                    num_changed+=1
            print(str(num_changed)+" "+str(qrs))

            new_random_text_joint = attacked_text.generate_new_attacked_text(random_text) 
            print ('attacked_text',attacked_text)
            print ('new_random_text_joint',new_random_text_joint)
            
            sim_remove_unnecessary_org, sim_remove_unnecessary_pert = self.sentence_encoder_use.encode([attacked_text.text, new_random_text_joint.text])

            if not isinstance(sim_remove_unnecessary_org, torch.Tensor):
                sim_remove_unnecessary_org = torch.tensor(sim_remove_unnecessary_org)

            if not isinstance(sim_remove_unnecessary_pert, torch.Tensor):
                sim_remove_unnecessary_pert = torch.tensor(sim_remove_unnecessary_pert)

            random_sim = self.sentence_encoder_use.sim_metric(sim_remove_unnecessary_org.unsqueeze(0), sim_remove_unnecessary_pert.unsqueeze(0))
            random_sim = random_sim.item()
            print ('random_sim1',random_sim)


            # random_sim = calc_sim(text_ls, [random_text], -1, sim_score_window, sim_predictor)[0]

            print ('qrs budget 1',qrs)
            if qrs > budget:
                # return fail 
                random_text_qrs_joint = attacked_text.generate_new_attacked_text(random_text) 

                sim_remove_unnecessary_org, sim_remove_unnecessary_pert = self.sentence_encoder_use.encode([attacked_text.text, random_text_qrs_joint.text])

                if not isinstance(sim_remove_unnecessary_org, torch.Tensor):
                    sim_remove_unnecessary_org = torch.tensor(sim_remove_unnecessary_org)

                if not isinstance(sim_remove_unnecessary_pert, torch.Tensor):
                    sim_remove_unnecessary_pert = torch.tensor(sim_remove_unnecessary_pert)

                failed_sem_sim = self.sentence_encoder_use.sim_metric(sim_remove_unnecessary_org.unsqueeze(0), sim_remove_unnecessary_pert.unsqueeze(0))
                failed_sem_sim = failed_sem_sim.item()
                print ('failed_sem_sim0',failed_sem_sim)

                if failed_sem_sim <  (1 - (self.similarity_threshold) / math.pi):
                    print ('returning failed result because best_sem_sim too low qrs too much', failed_sem_sim)
                    # results_inner, search_over = self.get_goal_results([attacked_text])
                    # return results_inner[0]
                    self.goal_function.model.reset_inference_steps()
                    return initial_result
                else:
                    print ('out of queries, random_text_qrs_joint',random_text_qrs_joint)
                    results_inner, search_over = self.get_goal_results([random_text_qrs_joint])
                    self.goal_function.model.reset_inference_steps()
                    return results_inner[0]
                
                

                 
                # return ' '.join(random_text), len(changed_indices), len(changed_indices), \
                #     orig_label, torch.argmax(predictor([random_text])), qrs, random_sim, random_sim

            if num_changed == 1:
                random_text_num_changed_joint = attacked_text.generate_new_attacked_text(random_text)  
                sim_remove_unnecessary_org, sim_remove_unnecessary_pert = self.sentence_encoder_use.encode([attacked_text.text, random_text_num_changed_joint.text])

                if not isinstance(sim_remove_unnecessary_org, torch.Tensor):
                    sim_remove_unnecessary_org = torch.tensor(sim_remove_unnecessary_org)

                if not isinstance(sim_remove_unnecessary_pert, torch.Tensor):
                    sim_remove_unnecessary_pert = torch.tensor(sim_remove_unnecessary_pert)

                failed_sem_sim = self.sentence_encoder_use.sim_metric(sim_remove_unnecessary_org.unsqueeze(0), sim_remove_unnecessary_pert.unsqueeze(0))
                failed_sem_sim = failed_sem_sim.item()
                print ('failed_sem_sim1',failed_sem_sim)

                if failed_sem_sim <  (1 - (self.similarity_threshold) / math.pi):
                    print ('returning failed result because best_sem_sim too low qrs too much', failed_sem_sim)
                    # results_inner, search_over = self.get_goal_results([attacked_text])
                    # return results_inner[0]
                    self.goal_function.model.reset_inference_steps()
                    return initial_result
                else:
                    print ('out of queries, random_text_num_changed_joint',random_text_num_changed_joint)
                    results_inner, search_over = self.get_goal_results([random_text_num_changed_joint])
                    self.goal_function.model.reset_inference_steps()
                    return results_inner[0]
                    
                # return failed
                # return ' '.join(random_text), 1, 1, \
                #     orig_label, torch.argmax(predictor([random_text])), qrs, random_sim, random_sim

            

            best_attack = random_text
            # best_sim = calc_sim(text_ls, [best_attack], -1, sim_score_window, sim_predictor)
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
                    # old_adv_embed.append([float(num) for num in embed_content[word_idx_dict[theta_old_text[idx]]].strip().split()[1:]])
                    old_adv_embed.append([float(num) for num in embed_content[word_idx_dict[theta_old_text[idx]]]])
                old_adv_embed_matrix = np.asarray(old_adv_embed)

                theta_old = old_adv_embed_matrix-words_perturb_embed_matrix
               
                u_vec = np.random.normal(loc=0.0, scale=1,size=theta_old.shape)
                theta_old_neighbor = theta_old+0.5*u_vec
                # print ('theta_old_neighbor',theta_old_neighbor)
                # Check if theta_old_neighbor is a 2D array
                if theta_old_neighbor.ndim != 2:
                    print('theta_old_neighbor not a 2D array. Skipping this iteration.')
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
                        # syn_feat = [float(num) for num in embed_content[word_idx_dict[syn]].strip().split()[1:]]
                        syn_feat = [float(num) for num in embed_content[word_idx_dict[syn]]]
                        syn_feat_set.append(syn_feat)

                    perturb_syn_dist = np.sum((syn_feat_set-perturb_target)**2, axis=1)
                    perturb_syn_order = np.argsort(perturb_syn_dist)
                    replacement = synonyms_all[perturb_word_idx][1][perturb_syn_order[0]]
                    
                    theta_old_neighbor_text[synonyms_all[perturb_word_idx][0]] = replacement

                    theta_old_neighbor_text_joint = attacked_text.generate_new_attacked_text(theta_old_neighbor_text)
                    print ('attacked_text',attacked_text)
                    print ('theta_old_neighbor_text_joint',theta_old_neighbor_text_joint)
                    if attacked_text.text == theta_old_neighbor_text_joint.text: # is word sub leads to perturbation being same as original sample skip
                        continue 
                    # elif theta_old_neighbor_text_joint in already_explored:
                    #     continue
                    # else:
                    #     already_explored.add(theta_old_neighbor_text_joint)
                    # model_outputs = self.goal_function._call_model([theta_old_neighbor_text_joint])
                    # current_goal_status = self.goal_function._get_goal_status(
                    #     model_outputs[0], theta_old_neighbor_text_joint, check_skip=False
                    # )
                    # self.number_of_queries+=1
                    results, search_over = self.get_goal_results([theta_old_neighbor_text_joint])
                    if search_over: 
                        self.goal_function.model.reset_inference_steps()
                        return initial_result
                    print ('qrs budget 2',qrs)
                    qrs+=1
                    self.number_of_queries+=1
                    self.goal_function.num_queries = self.number_of_queries
                    # self.goal_function.num_queries+=1
                    
                    if qrs > budget:
                        sim_remove_unnecessary_org, sim_remove_unnecessary_pert = self.sentence_encoder_use.encode([attacked_text.text, theta_old_neighbor_text_joint.text])

                        if not isinstance(sim_remove_unnecessary_org, torch.Tensor):
                            sim_remove_unnecessary_org = torch.tensor(sim_remove_unnecessary_org)

                        if not isinstance(sim_remove_unnecessary_pert, torch.Tensor):
                            sim_remove_unnecessary_pert = torch.tensor(sim_remove_unnecessary_pert)

                        failed_sem_sim = self.sentence_encoder_use.sim_metric(sim_remove_unnecessary_org.unsqueeze(0), sim_remove_unnecessary_pert.unsqueeze(0))
                        failed_sem_sim = failed_sem_sim.item()
                        print ('failed_sem_sim2',failed_sem_sim)

                        if failed_sem_sim <  (1 - (self.similarity_threshold) / math.pi): 
                            print ('returning failed result because best_sem_sim too low qrs too much', failed_sem_sim)
                            # results_inner, search_over = self.get_goal_results([attacked_text])
                            # return results_inner[0]
                            self.goal_function.model.reset_inference_steps()
                            return initial_result
                        else:
                            print ('out of queries, theta_old_neighbor_text_joint',theta_old_neighbor_text_joint)
                            # results_inner, search_over = self.get_goal_results([theta_old_neighbor_text_joint])
                            # return results_inner[0]
                            self.goal_function.model.reset_inference_steps()
                            return results[0]
                        
                        
                        # return results[0]   
                        # return ' '.join(best_attack), max_changes, len(changed_indices), \
                        #     orig_label, torch.argmax(predictor([best_attack])), qrs, sim, random_sim
                    if results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                        break



                    # pr = get_attack_result([theta_old_neighbor_text], predictor, orig_label, batch_size)
                    # qrs+=1

                    # if qrs > budget:
                    #     sim = best_sim[0]
                    #     max_changes = 0
                    #     for i in range(len(text_ls)):
                    #         if text_ls[i]!=best_attack[i]:
                    #             max_changes+=1

                    #     return ' '.join(best_attack), max_changes, len(changed_indices), \
                    #         orig_label, torch.argmax(predictor([best_attack])), qrs, sim, random_sim

                    # if np.sum(pr)>0:
                    #     break

                # if np.sum(pr)>0:

                if results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:

                     
                    theta_old_neighbor_text_joint = attacked_text.generate_new_attacked_text(theta_old_neighbor_text) 
                    

                    print ('attacked_text',attacked_text)
                    print ('theta_old_neighbor_text_joint',theta_old_neighbor_text_joint)
                    if attacked_text.text == theta_old_neighbor_text_joint.text: # is word sub leads to perturbation being same as original sample skip
                        continue 
                    # elif theta_old_neighbor_text_joint in already_explored: # if we already tried this perturbation try again
                    #     continue
                    # else:
                    #     already_explored.add(theta_old_neighbor_text_joint)
                    sim_remove_unnecessary_org, sim_remove_unnecessary_pert = self.sentence_encoder_use.encode([attacked_text.text, theta_old_neighbor_text_joint.text])

                    if not isinstance(sim_remove_unnecessary_org, torch.Tensor):
                        sim_remove_unnecessary_org = torch.tensor(sim_remove_unnecessary_org)

                    if not isinstance(sim_remove_unnecessary_pert, torch.Tensor):
                        sim_remove_unnecessary_pert = torch.tensor(sim_remove_unnecessary_pert)

                    sim_new = self.sentence_encoder_use.sim_metric(sim_remove_unnecessary_org.unsqueeze(0), sim_remove_unnecessary_pert.unsqueeze(0))
                    sim_new = sim_new.item()
                    print ('sim_new',sim_new)

                    # sim_new = calc_sim(text_ls, [theta_old_neighbor_text], -1, sim_score_window, sim_predictor)
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
                            # syn_feat = [float(num) for num in embed_content[word_idx_dict[syn]].strip().split()[1:]]
                            syn_feat = [float(num) for num in embed_content[word_idx_dict[syn]]]
                            syn_feat_set.append(syn_feat)

                        perturb_syn_dist = np.sum((syn_feat_set-perturb_target)**2, axis=1)
                        perturb_syn_order = np.argsort(perturb_syn_dist)
                        replacement = synonyms_all[perturb_word_idx][1][perturb_syn_order[0]]
                        
                        theta_new_text[synonyms_all[perturb_word_idx][0]] = replacement


                        theta_new_text_joint = attacked_text.generate_new_attacked_text(theta_new_text)
                        print ('attacked_text',attacked_text)
                        print ('theta_new_text_joint',theta_new_text_joint)
                        if attacked_text.text == theta_new_text_joint.text: # is word sub leads to perturbation being same as original sample skip
                            continue  
                        # elif theta_new_text_joint in already_explored: # if we already tried this perturbation try again
                        #     continue
                        # else:
                        #     already_explored.add(theta_new_text_joint)
                        results, search_over = self.get_goal_results([theta_new_text_joint])
                        if search_over: 
                            self.goal_function.model.reset_inference_steps()
                            return initial_result

                        qrs+=1
                        self.number_of_queries+=1
                        self.goal_function.num_queries = self.number_of_queries
                        
                        print ('qrs budget 3',qrs)
                        if qrs > budget:
                            sim_remove_unnecessary_org, sim_remove_unnecessary_pert = self.sentence_encoder_use.encode([attacked_text.text, theta_new_text_joint.text])

                            if not isinstance(sim_remove_unnecessary_org, torch.Tensor):
                                sim_remove_unnecessary_org = torch.tensor(sim_remove_unnecessary_org)

                            if not isinstance(sim_remove_unnecessary_pert, torch.Tensor):
                                sim_remove_unnecessary_pert = torch.tensor(sim_remove_unnecessary_pert)

                            failed_sem_sim = self.sentence_encoder_use.sim_metric(sim_remove_unnecessary_org.unsqueeze(0), sim_remove_unnecessary_pert.unsqueeze(0))
                            failed_sem_sim = failed_sem_sim.item()
                            print ('failed_sem_sim3',failed_sem_sim)

                            if failed_sem_sim <  (1 - (self.similarity_threshold) / math.pi):
                                print ('returning failed result because best_sem_sim too low qrs too much', failed_sem_sim)
                                # results_inner, search_over = self.get_goal_results([attacked_text])
                                # return results_inner[0]
                                self.goal_function.model.reset_inference_steps()
                                return initial_result
                            else:
                                # print ('out of queries, theta_new_text_joint',theta_new_text_joint)
                                # results_inner, search_over = self.get_goal_results([theta_new_text_joint])
                                # return results_inner[0]
                                self.goal_function.model.reset_inference_steps()
                                return results[0]
                            # return results[0]
                        if results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                            break


                        # pr = get_attack_result([theta_new_text], predictor, orig_label, batch_size)
                        # qrs+=1

                        # if qrs > budget:
                        #     sim = best_sim[0]
                        #     max_changes = 0
                        #     for i in range(len(text_ls)):
                        #         if text_ls[i]!=best_attack[i]:
                        #             max_changes+=1

                        #     return ' '.join(best_attack), max_changes, len(changed_indices), \
                        #         orig_label, torch.argmax(predictor([best_attack])), qrs, sim, random_sim

                        # if np.sum(pr)>0:
                        #     break
                    # if np.sum(pr)>0:
                    if results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                        theta_new_text_joint = attacked_text.generate_new_attacked_text(theta_new_text) 
                        print ('attacked_text',attacked_text)
                        print ('theta_new_text_joint',theta_new_text_joint)
                        if attacked_text.text == theta_new_text_joint.text: # is word sub leads to perturbation being same as original sample skip
                            continue
                        # elif theta_new_text_joint in already_explored: # if we already tried this perturbation try again
                        #     continue
                        # else:
                        #     already_explored.add(theta_new_text_joint)
                        sim_remove_unnecessary_org, sim_remove_unnecessary_pert = self.sentence_encoder_use.encode([attacked_text.text, theta_new_text_joint.text])

                        if not isinstance(sim_remove_unnecessary_org, torch.Tensor):
                            sim_remove_unnecessary_org = torch.tensor(sim_remove_unnecessary_org)

                        if not isinstance(sim_remove_unnecessary_pert, torch.Tensor):
                            sim_remove_unnecessary_pert = torch.tensor(sim_remove_unnecessary_pert)

                        sim_theta_new = self.sentence_encoder_use.sim_metric(sim_remove_unnecessary_org.unsqueeze(0), sim_remove_unnecessary_pert.unsqueeze(0))
                        sim_theta_new = sim_theta_new.item()
                        print ('sim_theta_new',sim_theta_new)
                        # sim_theta_new = calc_sim(text_ls, [theta_new_text], -1, sim_score_window, sim_predictor)
                        if sim_theta_new > best_sim:
                            best_attack = theta_new_text
                            best_sim = sim_theta_new

                    
                    # if np.sum(pr)>0:
                    if results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                        gamma_old_text = theta_new_text

                        gamma_old_text_joint = attacked_text.generate_new_attacked_text(gamma_old_text) 
                        print ('attacked_text',attacked_text)
                        print ('gamma_old_text_joint',gamma_old_text_joint)
                        if attacked_text.text == gamma_old_text_joint.text: # is word sub leads to perturbation being same as original sample skip
                            continue
                        # elif gamma_old_text_joint in already_explored: # if we already tried this perturbation try again
                        #     continue
                        # else:
                        #     already_explored.add(gamma_old_text_joint)
                        sim_remove_unnecessary_org, sim_remove_unnecessary_pert = self.sentence_encoder_use.encode([attacked_text.text, gamma_old_text_joint.text])

                        if not isinstance(sim_remove_unnecessary_org, torch.Tensor):
                            sim_remove_unnecessary_org = torch.tensor(sim_remove_unnecessary_org)

                        if not isinstance(sim_remove_unnecessary_pert, torch.Tensor):
                            sim_remove_unnecessary_pert = torch.tensor(sim_remove_unnecessary_pert)

                        gamma_sim_full = self.sentence_encoder_use.sim_metric(sim_remove_unnecessary_org.unsqueeze(0), sim_remove_unnecessary_pert.unsqueeze(0))
                        gamma_sim_full = gamma_sim_full.item()
                        print ('gamma_sim_full',gamma_sim_full)



                        # gamma_sim_full = calc_sim(text_ls, [gamma_old_text], -1, sim_score_window, sim_predictor)
                        gamma_old_adv_embed = []
                        for idx in words_perturb_doc_idx:
                            # gamma_old_adv_embed.append([float(num) for num in embed_content[word_idx_dict[gamma_old_text[idx]]].strip().split()[1:]])
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
                            print ('attacked_text',attacked_text)
                            print ('replaceback_text_joint',replaceback_text_joint)
                            if attacked_text.text == replaceback_text_joint.text: # is word sub leads to perturbation being same as original sample skip
                                continue
                            # elif replaceback_text_joint in already_explored: # if we already tried this perturbation try again
                            #     continue
                            # else:
                            #     already_explored.add(replaceback_text_joint)
                            sim_remove_unnecessary_org, sim_remove_unnecessary_pert = self.sentence_encoder_use.encode([attacked_text.text, replaceback_text_joint.text])

                            if not isinstance(sim_remove_unnecessary_org, torch.Tensor):
                                sim_remove_unnecessary_org = torch.tensor(sim_remove_unnecessary_org)

                            if not isinstance(sim_remove_unnecessary_pert, torch.Tensor):
                                sim_remove_unnecessary_pert = torch.tensor(sim_remove_unnecessary_pert)

                            replaceback_sims = self.sentence_encoder_use.sim_metric(sim_remove_unnecessary_org.unsqueeze(0), sim_remove_unnecessary_pert.unsqueeze(0))
                            replaceback_sims = replaceback_sims.item()
                            print ('replaceback_sims',replaceback_sims)

                            # replaceback_sims = calc_sim(text_ls, [replaceback_text], -1, sim_score_window, sim_predictor)
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
                                # syn_feat = [float(num) for num in embed_content[word_idx_dict[syn]].strip().split()[1:]]
                                syn_feat = [float(num) for num in embed_content[word_idx_dict[syn]]]
                                syn_feat_set.append(syn_feat)

                            perturb_syn_dist = np.sum((syn_feat_set-perturb_target)**2, axis=1)
                            perturb_syn_order = np.argsort(perturb_syn_dist)
                            replacement = synonyms_all[perturb_word_idx][1][perturb_syn_order[0]]
                            
                            theta_new_text[synonyms_all[perturb_word_idx][0]] = replacement


                            theta_new_text_joint = attacked_text.generate_new_attacked_text(theta_new_text)
                            print ('attacked_text',attacked_text)
                            print ('theta_new_text_joint',theta_new_text_joint)
                            if attacked_text.text == theta_new_text_joint.text: # is word sub leads to perturbation being same as original sample skip
                                continue 
                            # elif theta_new_text_joint in already_explored: # if we already tried this perturbation try again
                            #     continue
                            # else:
                            #     already_explored.add(theta_new_text_joint)
                            # model_outputs = self.goal_function._call_model([theta_new_text_joint])
                            # current_goal_status = self.goal_function._get_goal_status(
                            #     model_outputs[0], theta_new_text_joint, check_skip=False
                            # )
                            # self.number_of_queries+=1
                            results, search_over = self.get_goal_results([theta_new_text_joint])
                            if search_over:
                                self.goal_function.model.reset_inference_steps() 
                                return initial_result

                            qrs+=1 
                            self.number_of_queries+=1
                            self.goal_function.num_queries = self.number_of_queries
                            # self.goal_function.num_queries+=1
                            print ('qrs budget 4',qrs)
                            if qrs > budget:
                                sim_remove_unnecessary_org, sim_remove_unnecessary_pert = self.sentence_encoder_use.encode([attacked_text.text, theta_new_text_joint.text])

                                if not isinstance(sim_remove_unnecessary_org, torch.Tensor):
                                    sim_remove_unnecessary_org = torch.tensor(sim_remove_unnecessary_org)

                                if not isinstance(sim_remove_unnecessary_pert, torch.Tensor):
                                    sim_remove_unnecessary_pert = torch.tensor(sim_remove_unnecessary_pert)

                                failed_sem_sim = self.sentence_encoder_use.sim_metric(sim_remove_unnecessary_org.unsqueeze(0), sim_remove_unnecessary_pert.unsqueeze(0))
                                failed_sem_sim = failed_sem_sim.item()
                                print ('failed_sem_sim4',failed_sem_sim)

                                if failed_sem_sim <  (1 - (self.similarity_threshold) / math.pi):
                                    print ('returning failed result because best_sem_sim too low qrs too much', failed_sem_sim)
                                    # results_inner, search_over = self.get_goal_results([attacked_text])
                                    # return results_inner[0]
                                    self.goal_function.model.reset_inference_steps()
                                    return initial_result
                                else:
                                    print ('out of queries, theta_new_text_joint',theta_new_text_joint)
                                    # results_inner, search_over = self.get_goal_results([theta_new_text_joint])
                                    # return results_inner[0]
                                    self.goal_function.model.reset_inference_steps()
                                    return results[0]
                                # sim = best_sim[0]
                                # max_changes = 0
                                # for i in range(len(text_ls)):
                                #     if text_ls[i]!=best_attack[i]:
                                #         max_changes+=1

                                # return results[0]
                                # return ' '.join(best_attack), max_changes, len(changed_indices), \
                                #     orig_label, torch.argmax(predictor([best_attack])), qrs, sim, random_sim
                            # if current_goal_status != GoalFunctionResultStatus.SUCCEEDED:
                            #     break
                            if results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                                break

                            # pr = get_attack_result([theta_new_text], predictor, orig_label, batch_size)
                            
                            
                            # qrs+=1

                            # if qrs > budget:
                            #     sim = best_sim[0]
                            #     max_changes = 0
                            #     for i in range(len(text_ls)):
                            #         if text_ls[i]!=best_attack[i]:
                            #             max_changes+=1

                            #     return ' '.join(best_attack), max_changes, len(changed_indices), \
                            #         orig_label, torch.argmax(predictor([best_attack])), qrs, sim, random_sim

                            # if np.sum(pr)>0:
                            #     break

                    
                        # if np.sum(pr)>0:
                        if results[0].goal_status == GoalFunctionResultStatus.SUCCEEDED:
                            theta_new_text_joint = attacked_text.generate_new_attacked_text(theta_new_text) 
                            print ('attacked_text',attacked_text)
                            print ('theta_new_text_joint 2',theta_new_text_joint)
                            if attacked_text.text == theta_new_text_joint.text: # is word sub leads to perturbation being same as original sample skip
                                continue
                            # elif theta_new_text_joint in already_explored: # if we already tried this perturbation try again
                            #     continue
                            # else:
                            #     already_explored.add(theta_new_text_joint)
                            sim_remove_unnecessary_org, sim_remove_unnecessary_pert = self.sentence_encoder_use.encode([attacked_text.text, theta_new_text_joint.text])

                            if not isinstance(sim_remove_unnecessary_org, torch.Tensor):
                                sim_remove_unnecessary_org = torch.tensor(sim_remove_unnecessary_org)

                            if not isinstance(sim_remove_unnecessary_pert, torch.Tensor):
                                sim_remove_unnecessary_pert = torch.tensor(sim_remove_unnecessary_pert)

                            sim_theta_new = self.sentence_encoder_use.sim_metric(sim_remove_unnecessary_org.unsqueeze(0), sim_remove_unnecessary_pert.unsqueeze(0))
                            sim_theta_new = sim_theta_new.item()
                            print ('sim_theta_new',sim_theta_new)
                            # sim_theta_new = calc_sim(text_ls, [theta_new_text], -1, sim_score_window, sim_predictor)
                            if sim_theta_new > best_sim:
                                best_attack = theta_new_text
                                best_sim = sim_theta_new


            best_attack_joint = attacked_text.generate_new_attacked_text(best_attack)
            


            print ('attacked_text',attacked_text)
            print ('best_attack_joint',best_attack_joint)
            if attacked_text.text == best_attack_joint.text: # is word sub leads to perturbation being same as original sample skip
                self.goal_function.model.reset_inference_steps() 
                return initial_result # in this case i return the initial result, shouldent really ever happen since we always check if the perturbed sample is the same as original one at each step and ignore perturbation if it is.
            
            sim_remove_unnecessary_org, sim_remove_unnecessary_pert = self.sentence_encoder_use.encode([attacked_text.text, best_attack_joint.text])

            if not isinstance(sim_remove_unnecessary_org, torch.Tensor):
                sim_remove_unnecessary_org = torch.tensor(sim_remove_unnecessary_org)

            if not isinstance(sim_remove_unnecessary_pert, torch.Tensor):
                sim_remove_unnecessary_pert = torch.tensor(sim_remove_unnecessary_pert)

            best_sem_sim = self.sentence_encoder_use.sim_metric(sim_remove_unnecessary_org.unsqueeze(0), sim_remove_unnecessary_pert.unsqueeze(0))
            best_sem_sim = best_sem_sim.item()
            print ('best_sem_sim',best_sem_sim)

            if best_sem_sim <  (1 - (self.similarity_threshold) / math.pi):
                print ('returning failed result because best_sem_sim too low', best_sem_sim)
                # results, search_over = self.get_goal_results([attacked_text])
                # return results[0]
                self.goal_function.model.reset_inference_steps()
                return initial_result



            print ('best sim meets threshod,best_attack_joint',best_attack_joint)
            results, search_over = self.get_goal_results([best_attack_joint])
            self.number_of_queries+=1
            self.goal_function.num_queries = self.number_of_queries
            # self.goal_function.num_queries+=1
            if search_over: 
                self.goal_function.model.reset_inference_steps() 
                return initial_result
            print ('results best attack',results)
            self.goal_function.model.reset_inference_steps()
            return results[0]

            # sim = best_sim[0]
            # print ('last sim',sim)
            # max_changes = 0
            # for i in range(len(text_ls)):
            #     if text_ls[i]!=best_attack[i]:
            #         max_changes+=1
            # print ('best_attack',best_attack)
            
            # print ('return everything ',' '.join(best_attack), max_changes, len(changed_indices),  orig_label, torch.argmax(predictor([best_attack])), qrs, sim, random_sim)
            
            # sys.exit()
            # return ' '.join(best_attack), max_changes, len(changed_indices), \
            #       orig_label, torch.argmax(predictor([best_attack])), qrs, sim, random_sim

            

        else:
            print ('returning failed result because flag==0')
            # results, search_over = self.get_goal_results([attacked_text])
            # return results[0]
            self.goal_function.model.reset_inference_steps()
            return initial_result
            # print("Not Found")
            # return '', 0,0, orig_label, orig_label, 0, 0, 0
        
        
        sys.exit()
        # Att_sen_new_sentence = AttackedText(new_sentence) 
        # print ('att sen new indices',Att_sen_new_sentence,Att_sen_new_sentence.attack_attrs)
        # print ('att sen new words',Att_sen_new_sentence.words, len(Att_sen_new_sentence.words) ) 
        # Att_sen_new_sentence.attack_attrs['newly_modified_indices'] = {0}
        # Att_sen_new_sentence.attack_attrs["previous_attacked_text"] = current_text
        # # Att_sen_new_sentence.attack_attrs['modified_indices'] = set(range(len(Att_sen_new_sentence.words)))
        # # Att_sen_new_sentence.attack_attrs['original_index_map'] = original_index_map
        # Att_sen_new_sentence.attack_attrs['modified_indices'] = set(Att_sen_new_sentence.attack_attrs['original_index_map'])
        # print ('att sen new indices2',Att_sen_new_sentence,Att_sen_new_sentence.attack_attrs)

        # attacked_text.generate_new_attacked_text(new_words)




        
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
                print ('sim_score',sim_score, (1 - (args.similarity_threshold) / math.pi))
                if sim_score <  (1 - (args.similarity_threshold) / math.pi):
                    continue

                # final_result.num_queries = self.number_of_queries
                self.goal_function.num_queries = self.number_of_queries
                print ('final_result.num_queries',final_result.num_queries)
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
            print ('words_perturb',words_perturb)
            print ('start_i',start_i)
            words = tmp_text.words
            pos_tags = nltk.pos_tag(words)   
            print ('pos_tags nltk',pos_tags)
            if pos_tags[start_i][1].startswith(('VB', 'NN', 'JJ', 'RB')): 
                # print ('pos_tags[start_i][1]',pos_tags[start_i][1])
                replaced_with_synonyms = self.get_transformations(tmp_text, original_text=tmp_text,indices_to_modify=[start_i])
                print ('replaced_with_synonyms should be 10?',replaced_with_synonyms,len(replaced_with_synonyms))
                
                if replaced_with_synonyms:
                    tmp_text = random.choice(replaced_with_synonyms)
                else:
                    pass
                
            start_i+=1
            
        adv_text = tmp_text
        return adv_text

 

    # def remove_unnecessary_words(self, perturbed_text, original_text, check_skip=False):
    #     # Step 1: Identify words to replace back
    #     candidate_set = []
    #     word_importance_scores = [] 
    #     # print ('original_text',original_text)
    #     # print ('perturbed_text',perturbed_text)
    #     for i, (perturbed_word, original_word) in enumerate(zip(perturbed_text.words, original_text.words)):
    #         if perturbed_word != original_word:
    #             # Replace perturbed_word with original_word
    #             temp_text = perturbed_text.replace_word_at_index(i, original_word)

    #             # Step 2: Check if still adversarial and calculate semantic similarity
    #             model_outputs = self.goal_function._call_model([temp_text])
    #             current_goal_status = self.goal_function._get_goal_status(
    #                 model_outputs[0], temp_text, check_skip=check_skip
    #             )
    #             self.number_of_queries+=1
    #             # print ('temp_text',temp_text,i,current_goal_status,GoalFunctionResultStatus.SUCCEEDED)
    #             if current_goal_status == GoalFunctionResultStatus.SUCCEEDED:
    #                 candidate_set.append((i, temp_text))
    #                 sim_remove_unnecessary_org, sim_remove_unnecessary_pert = self.sentence_encoder_use.encode([original_text.text, temp_text.text])

    #                 if not isinstance(sim_remove_unnecessary_org, torch.Tensor):
    #                     sim_remove_unnecessary_org = torch.tensor(sim_remove_unnecessary_org)

    #                 if not isinstance(sim_remove_unnecessary_pert, torch.Tensor):
    #                     sim_remove_unnecessary_pert = torch.tensor(sim_remove_unnecessary_pert)

    #                 sim_score = self.sentence_encoder_use.sim_metric(sim_remove_unnecessary_org.unsqueeze(0), sim_remove_unnecessary_pert.unsqueeze(0))
                    
    #                 word_importance_scores.append((i, sim_score))

    #     # Step 3: Sort word importance scores in descending order and restore original words
    #     word_importance_scores.sort(key=lambda x: x[1], reverse=True)
    #     print ('attack_attrs ',perturbed_text.attack_attrs,perturbed_text  ) 
    #     print ('replace indexs',word_importance_scores)
    #     for idx, _ in word_importance_scores:
    #         temp_text2 = perturbed_text.replace_word_at_index(idx, original_text.words[idx])
    #         temp_text2.attack_attrs['modified_indices'].remove(idx)
    #         print ('temp_text2_word_imp',idx,temp_text2.attack_attrs,temp_text2)
    #         # print ('original_index_map',temp_text2.attack_attrs.original_index_map)
            
    #         model_outputs = self.goal_function._call_model([temp_text2])
    #         current_goal_status = self.goal_function._get_goal_status(
    #             model_outputs[0], temp_text2, check_skip=check_skip
    #         )
    #         self.number_of_queries+=1

    #         # print ('temp_text2',temp_text2,current_goal_status, GoalFunctionResultStatus.SUCCEEDED)

 
    #         if current_goal_status == GoalFunctionResultStatus.SUCCEEDED:
    #             # If perturbed_text is no longer adversarial, revert the last change
    #             perturbed_text = temp_text2
    #             # perturbed_text = perturbed_text.replace_word_at_index(idx, perturbed_text.words[idx])
    #         else:
    #             break
    #     # print ('original_text',original_text)
    #     # print ('perturbed_text',perturbed_text) 
    #     return perturbed_text

    # def get_vector(self, embedding, word):
    #     if isinstance(word, str):
    #         if word in embedding._word2index:
    #             word_index = embedding._word2index[word]
    #         else:
    #             return None  # Word not found in the dictionary
    #     else:
    #         word_index = word

    #     vector = embedding.embedding_matrix[word_index]
    #     return torch.tensor(vector).to(textattack.shared.utils.device)

    # def push_words_towards_original(self, perturbed_text, original_text, check_skip=False):
    #     # Step 1: Calculate Euclidean distances and sampling probabilities
    #     distances = []
    #     for i, (perturbed_word, original_word) in enumerate(zip(perturbed_text.words, original_text.words)):
    #         if perturbed_word != original_word:
    #             # Using the get_vector function
    #             perturbed_vec = self.get_vector(self.embedding, perturbed_word)
    #             if perturbed_vec is None:
    #                 continue  # Skip to the next word
    #             original_vec = self.get_vector(self.embedding, original_word)
    #             if original_vec is None:
    #                 continue  # Skip to the next word
    #             distance = np.linalg.norm(perturbed_vec.cpu().numpy() - original_vec.cpu().numpy())
    #             distances.append((i, distance))

    #     if not distances:
    #         return perturbed_text

    #     # Normalize distances to get probabilities
    #     distances.sort(key=lambda x: x[1])
    #     indices, dist_values = zip(*distances)
    #     exp_dist_values = np.exp(dist_values)
    #     probabilities = exp_dist_values / np.sum(exp_dist_values)
    #     print ('probabilities',probabilities)

    #     # temp_perturbed_text = copy.deepcopy(perturbed_text)
        
    #     # Step 2: Iterate with sampling based on the probabilities 
    #     while len(indices) > 0:
    #         i = np.random.choice(indices, p=probabilities)
    #         print ('indices',indices,i)
    #         perturbed_word = perturbed_text.words[i]
    #         original_word = original_text.words[i]

    #         sentence_replaced = self.get_transformations(original_text, original_text=original_text, indices_to_modify=[i])
    #         synonyms = [s.words[i] for s in sentence_replaced]


    #         # Get top k synonyms
    #         k = 10  # Number of synonyms to sample
    #         top_k_synonyms_indexes  = self.embedding.nearest_neighbours(self.embedding._word2index[original_word], topn=k)
    #         top_k_synonyms = [self.embedding._index2word[index] for index in top_k_synonyms_indexes]

    #         # Find the best anchor synonym with the highest semantic similarity
    #         max_similarity = -float('inf')
    #         w_bar = None
    #         temp_text_bar = None
    #         filtered_synonyms = None
    #         print ('top_k_synonyms',top_k_synonyms)
    #         # temp_text2 = copy.deepcopy(perturbed_text)
    #         for synonym in top_k_synonyms:
    #             if perturbed_word == synonym:
    #                 continue # skip swapping the same word
    #             print ('synonym',i,synonym)
    #             # temp_text2 = copy.deepcopy(perturbed_text)
    #             temp_text2 = perturbed_text.replace_word_at_index(i, synonym)

    #             # Check if the substitution still results in an adversarial example
    #             model_outputs = self.goal_function._call_model([temp_text2])
    #             current_goal_status = self.goal_function._get_goal_status(
    #                 model_outputs[0], temp_text2, check_skip=check_skip
    #             )
    #             self.number_of_queries+=1

    #             print ('temp_text2_top_k_syn',i,synonym,temp_text2.attack_attrs,temp_text2,current_goal_status , GoalFunctionResultStatus.SUCCEEDED)

    #             if current_goal_status == GoalFunctionResultStatus.SUCCEEDED:
    #                 # Compute semantic similarity at the word level
    #                 sim_remove_unnecessary_org, sim_remove_unnecessary_pert = self.sentence_encoder_use.encode([original_word, synonym])

    #                 if not isinstance(sim_remove_unnecessary_org, torch.Tensor):
    #                     sim_remove_unnecessary_org = torch.tensor(sim_remove_unnecessary_org)

    #                 if not isinstance(sim_remove_unnecessary_pert, torch.Tensor):
    #                     sim_remove_unnecessary_pert = torch.tensor(sim_remove_unnecessary_pert)

    #                 sim_score = self.sentence_encoder_use.sim_metric(sim_remove_unnecessary_org.unsqueeze(0), sim_remove_unnecessary_pert.unsqueeze(0)).item()
    #                 print ('sim scores push towards orgin')
    #                 if sim_score > max_similarity:
    #                     max_similarity = sim_score
    #                     w_bar = synonym
    #                     temp_text_bar = temp_text2

            
    #         if w_bar:# is None:
    #               # Skip this index if no suitable anchor synonym is found
            
    #             number_entries = len(self.embedding.nn_matrix[self.embedding._word2index[original_word]] )
    #             print ('num entries',number_entries)
    #             all_synonyms = self.embedding.nearest_neighbours(self.embedding._word2index[original_word], topn=number_entries)
    #             all_synonyms = [self.embedding._index2word[index] for index in all_synonyms]
            
    #             print ('all_synonyms',all_synonyms)
    #             filtered_synonyms = []
    #             for synonym in all_synonyms:
    #                 if perturbed_word == synonym or w_bar == synonym  :
    #                     continue # skip swapping/checking the same word and the anchor word
    #                 # Compute semantic similarity with w_bar and original_word
    #                 sim_w_bar, sim_synonym = self.sentence_encoder_use.encode([w_bar, synonym])
    #                 sim_org, sim_synonym_org = self.sentence_encoder_use.encode([original_word, synonym])

    #                 if not isinstance(sim_w_bar, torch.Tensor):
    #                     sim_w_bar = torch.tensor(sim_w_bar)
    #                 if not isinstance(sim_synonym, torch.Tensor):
    #                     sim_synonym = torch.tensor(sim_synonym)
    #                 if not isinstance(sim_org, torch.Tensor):
    #                     sim_org = torch.tensor(sim_org)
    #                 if not isinstance(sim_synonym_org, torch.Tensor):
    #                     sim_synonym_org = torch.tensor(sim_synonym_org)

    #                 sim_score_w_bar = self.sentence_encoder_use.sim_metric(sim_w_bar.unsqueeze(0), sim_synonym.unsqueeze(0)).item()
    #                 sim_score_org = self.sentence_encoder_use.sim_metric(sim_org.unsqueeze(0), sim_synonym_org.unsqueeze(0)).item()

    #                 if sim_score_w_bar > sim_score_org:
    #                     filtered_synonyms.append((sim_score_w_bar, synonym))

    #         if  filtered_synonyms:
    #             # continue  # Skip this index if no suitable synonym is found

    #             # Sort the filtered synonyms by their semantic similarity score in descending order
    #             filtered_synonyms.sort(key=lambda item: item[0], reverse=True)
    #             print ('filtered_synonyms',filtered_synonyms)
                
                

    #             print ('perturbed text',perturbed_text.attack_attrs,perturbed_text)
    #             for _, synonym in filtered_synonyms:
    #                 temp_text2 = perturbed_text.replace_word_at_index(i, synonym) 
    #                 # temp_text2.attack_attrs['modified_indices'].remove(i)
    #                 print ('temp_text2_filtered_syn',i,temp_text2.attack_attrs,temp_text2)
    #                 # Check if the substitution still results in an adversarial example
    #                 model_outputs = self.goal_function._call_model([temp_text2]) 
    #                 current_goal_status = self.goal_function._get_goal_status(
    #                     model_outputs[0], temp_text2, check_skip=check_skip
    #                 )
    #                 self.number_of_queries+=1
    #                 print ('temp_text2',temp_text2,current_goal_status, GoalFunctionResultStatus.SUCCEEDED)
    #                 if current_goal_status == GoalFunctionResultStatus.SUCCEEDED:
    #                     perturbed_text = temp_text2
    #                     break
                    
            
    #         print ('perturbed_text',perturbed_text)  
    #         idx = indices.index(i)
    #         indices = indices[:idx] + indices[idx + 1:] 
    #         print ('indices2',indices,idx)
            
    #         probabilities = np.delete(probabilities, idx)
    #         probabilities /= np.sum(probabilities)   
    #     # sys.exit()
    #     return perturbed_text 

    def get_transformations(self, text, index):
        return self.transformation(text, index)

    def get_similarity(self, word1, word2):
        return self.transformation.get_cosine_similarity(word1, word2)
    
    @property
    def is_black_box(self):
        return True
