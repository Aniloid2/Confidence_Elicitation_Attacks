
from textattack.datasets import HuggingFaceDataset
# from datasets import load_dataset 
def load_data(args):
    from datasets import load_dataset 
    task = args.task
    if task == 'sst2': 
        print(f"Loading SST2 dataset from 'stanfordnlp/sst2'...")
        strategy_dataset = load_dataset('stanfordnlp/sst2', split='validation')#.select(range(500)) # example: taking 50 samples
        print(f"SST2 dataset loaded successfully", type(strategy_dataset), 'column names', strategy_dataset.column_names)
        # sys.exit()
        # strategy_dataset = strategy_dataset.rename_column("sentence", "text")
        # strategy_dataset = strategy_dataset.rename_column("label", "label")
        per_class_samples = args.num_examples//args.n_classes
        dataset = HuggingFaceDataset(strategy_dataset, split="test", shuffle=True)
        # for i,j in dataset:
        #     print ('i',i,j)
        # sys.exit()

        # dataset = HuggingFaceDataset("glue", "sst2", split="validation", shuffle=True)
        label_names = dataset.label_names

        # For dataset_class_1, only include sentences with 3 or more characters and label == 1
        dataset_class_1 = [(text['sentence'], label) for (text, label) in dataset if label == 1 and len(text['sentence']) >= 3]
        dataset_class_1_t = dataset_class_1[:per_class_samples]
        incontext_dataset_class_1 = dataset_class_1[-5:]

        # For dataset_class_0, only include sentences with 3 or more characters and label == 0
        dataset_class_0 = [(text['sentence'], label) for (text, label) in dataset if label == 0 and len(text['sentence']) >= 3]
        dataset_class_0_t = dataset_class_0[:per_class_samples]
        incontext_dataset_class_0 = dataset_class_0[-5:]

        dataset_class = dataset_class_1_t + dataset_class_0_t
        return dataset_class, label_names

    elif task == 'ag_news':
        print(f"Loading Ag News dataset from 'fancyzhx/ag_news'...")
        strategy_dataset = load_dataset('fancyzhx/ag_news', split='test')#.select(range(500)) # example: taking 50 samples
        print(f"Ag News dataset loaded successfully", type(strategy_dataset), 'column names', strategy_dataset.column_names)
        
        dataset = HuggingFaceDataset(strategy_dataset, split="test", shuffle=True)
        # dataset = HuggingFaceDataset('ag_news', split="test", shuffle=True)
        label_names = dataset.label_names
        per_class_samples = args.num_examples//args.n_classes
        # For dataset_class_1, only include documents with 3 or more characters and label == 1
        dataset_class_1 = [(text['text'], label) for (text, label) in dataset if label == 1 and len(text['text']) >= 3]
        dataset_class_1_t = dataset_class_1[:per_class_samples]
        incontext_dataset_class_1 = dataset_class_1[-5:]

        # For dataset_class_0, only include documents with 3 or more characters and label == 0
        dataset_class_0 = [(text['text'], label) for (text, label) in dataset if label == 0 and len(text['text']) >= 3]
        dataset_class_0_t = dataset_class_0[:per_class_samples]
        incontext_dataset_class_0 = dataset_class_0[-5:]

        # For dataset_class_2
        dataset_class_2 = [(text['text'], label) for (text, label) in dataset if label == 2 and len(text['text']) >= 3]
        dataset_class_2_t = dataset_class_2[:per_class_samples]
        incontext_dataset_class_2 = dataset_class_2[-5:]

        # For dataset_class_3
        dataset_class_3 = [(text['text'], label) for (text, label) in dataset if label == 3 and len(text['text']) >= 3]
        dataset_class_3_t = dataset_class_3[:per_class_samples]
        incontext_dataset_class_3 = dataset_class_3[-5:]

        # Combine datasets from different classes
        dataset_class = dataset_class_0_t + dataset_class_1_t + dataset_class_2_t + dataset_class_3_t

        print(f'Total filtered dataset size: {len(dataset_class)}')
        print(f'In-context samples for class 1: {incontext_dataset_class_1}')
        print(f'In-context samples for class 0: {incontext_dataset_class_0}')
        print(f'In-context samples for class 2: {incontext_dataset_class_2}')
        print(f'In-context samples for class 3: {incontext_dataset_class_3}')
        print (dataset_class)
        return dataset_class, label_names 

    elif task == 'popQA':
        from datasets import load_dataset
        import json
        from collections import defaultdict

        split = "test"
        print(f"Loading PopQA {split} dataset from 'akariasai/PopQA'...")
        raw_dataset = load_dataset("akariasai/PopQA", split=split).select(range(5))
        print(f"PopQA {split} dataset loaded successfully", type(raw_dataset),'column names',raw_dataset.column_names)

        # add a new column, and for each datasample given raw_dataset['prop'] (e.g capital, business etc) extract all other rows that have the same raw_dataset['prop'] and add their answers which are accessed raw_dataset['possible_answers'] and add them to a new column raw_dataset['wrong_possible_answers'] 
        # Create a dictionary to hold all possible answers for each 'prop'
        # prop_to_answers = defaultdict(list)
        # # Populate the dictionary
        # for sample in raw_dataset:
        #     prop_to_answers[sample['prop']].append(sample['possible_answers'])


        prop_to_answers = {}
        # Populate the dictionary
        for sample in raw_dataset:
            # print ('sample',sample)
            if sample['prop'] in prop_to_answers:
                prop_to_answers[sample['prop']] |= set(json.loads(sample['possible_answers']))
            else:
                prop_to_answers[sample['prop']] = set(json.loads(sample['possible_answers']))
                
 

        # Create the new column 'wrong_possible_answers'
        wrong_possible_answers_col = []
        possible_correct_incorrect_answers_col = []
        for sample in raw_dataset:
            current_prop = sample['prop']
            # current_answers = sample['possible_answers']
            current_answers = set(json.loads(sample['possible_answers']))
            print ('current_answers',current_answers)

            # Get all possible answers for the current 'prop'
            # all_answers_for_prop = prop_to_answers[current_prop]
            all_answers_for_prop =  prop_to_answers[current_prop]
            print ('all_answers_for_prop',all_answers_for_prop)

            # Perform set difference to find wrong possible answers
            wrong_answers = all_answers_for_prop - current_answers
            print ('wrong_answers',wrong_answers)
            # Convert back to list
            wrong_possible_answers_col.append(json.dumps(list(wrong_answers))) 

            possible_correct_incorrect_answers_col.append(json.dumps({
                'correct': list(current_answers),
                'incorrect': list(wrong_answers)
            }))
            # # Create the list of wrong possible answers by excluding the current sample's answers
            # wrong_answers = [ans for ans in all_answers_for_prop if ans != current_answers]

            # # Flatten the list of lists
            # wrong_answers = [item for sublist in wrong_answers for item in sublist]

            # wrong_possible_answers_col.append(wrong_answers) 
        # Add the new column to the dataset
        raw_dataset = raw_dataset.add_column("wrong_possible_answers", wrong_possible_answers_col)
        raw_dataset = raw_dataset.add_column("possible_correct_incorrect_answers", possible_correct_incorrect_answers_col)

        print ('raw_dataset',raw_dataset)
        print (raw_dataset['wrong_possible_answers']) 
        print (raw_dataset['possible_correct_incorrect_answers']) 
        # with   create all possible answers, in possible_answers, we insert a dictionary = {'correct':{},'incorrect':{}} 
        raw_dataset = raw_dataset.rename_column("possible_correct_incorrect_answers", "label")
        raw_dataset = raw_dataset.rename_column("question", "text") 
        # Create the PyTorch dataset
        dataset = HuggingFaceDataset(raw_dataset,split="test", shuffle=True)
        print(f"Converted to PyTorch Dataset", type(dataset))
        label_names = None
 
        # Collect data for dataset_class_t and incontext_dataset_class
        dataset_class = [] 
        dataset_class = [(text['text'], label) for (text, label) in dataset]
        dataset_class_t = dataset_class[:args.num_examples]
        incontext_dataset_class = dataset_class[-5:]

        print(f'Total filtered dataset size for PopQA: {len(dataset_class)}')
        print(f'In-context samples for PopQA: {incontext_dataset_class}')
        
        
        return dataset_class_t, label_names
    elif task == 'strategyQA':
        from datasets import load_dataset 
        print(f"Loading StrategyQA dataset from 'ChilleD/StrategyQA'...")
        strategy_dataset = load_dataset('ChilleD/StrategyQA', split='test').select(range(500)) # example: taking 50 samples
        print(f"StrategyQA dataset loaded successfully", type(strategy_dataset), 'column names', strategy_dataset.column_names)

        strategy_dataset = strategy_dataset.rename_column("question", "text")
        strategy_dataset = strategy_dataset.rename_column("answer", "label")

        strategy_dataset = HuggingFaceDataset(strategy_dataset, split="test", shuffle=True)
        print(f"Converted StrategyQA to PyTorch Dataset", type(strategy_dataset))

        strategy_dataset_class = []
        strategy_dataset_class = [(text['text'], int(label)) for (text, label) in strategy_dataset]
        strategy_dataset_class_t = strategy_dataset_class[:args.num_examples]#[17:18]
        strategy_incontext_dataset_class = strategy_dataset_class[-5:]
        label_names = ['false','true']
        print(f'Total filtered dataset size for StrategyQA: {len(strategy_dataset)}')
        print(f'In-context samples for StrategyQA: {strategy_incontext_dataset_class}')
        return strategy_dataset_class_t, label_names
     
    else:
        print("Task not supported.")


from textattack.datasets import Dataset

class SimpleDataset(Dataset):
    def __init__(self, examples, label_names=None):
        """
        args:
            examples: list of tuples where each tuple is (text, label)
            label_names: list of strings representing label names (optional)
        """
        self.examples = examples
        self.label_names = label_names
        self.shuffled = False  # Set to True if you shuffle the examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]