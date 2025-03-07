

import os

DATA_LOCATION = {
    'sst2': 'stanfordnlp/sst2',
    'ag_news': 'fancyzhx/ag_news',
    'strategyQA': 'ChilleD/StrategyQA',
    'popQA': 'akariasai/PopQA',
    'triviaQA': 'mandarjoshi/trivia_qa',
    'mnli': 'nyu-mll/multi_nli',
    'rte': ('glue', 'rte'),
    'qqp': ('glue', 'qqp'),
    'qnli': ('glue', 'qqp'),
}

def load_huggingface_dataset(name, split, per_class_samples):
        from textattack.datasets import HuggingFaceDataset
        from datasets import load_dataset 
        dataset = load_dataset(name,split, split='validation')
        print(f"{name} dataset loaded successfully", type(dataset), 'column names', dataset.column_names)
        
        dataset = HuggingFaceDataset(dataset, split=split, shuffle=True)
        label_names = dataset.label_names if hasattr(dataset, 'label_names') else None
        return dataset, label_names

def load_data(args):
    
    task = args.task
    dataset_location = DATA_LOCATION[args.task] 
    if task == 'sst2': 
        from textattack.datasets import HuggingFaceDataset
        from datasets import load_dataset  
        print(f"Loading {args.task} dataset from {dataset_location}...")
        args.ceattack_logger.info(f"Loading {args.task} dataset from {dataset_location}...")
        strategy_dataset = load_dataset(dataset_location, split='validation')#.select(range(500)) # example: taking 50 samples
        print(f"{args.task} dataset loaded successfully", type(strategy_dataset), 'column names', strategy_dataset.column_names)
        args.ceattack_logger.info(f"{args.task} dataset loaded successfully {type(strategy_dataset)} column {strategy_dataset.column_names}")
        
        per_class_samples = args.num_examples//args.n_classes
        dataset = HuggingFaceDataset(strategy_dataset, split="test", shuffle=True)
        
        label_names = dataset.label_names

        dataset_class_1 = [(text['sentence'], label) for (text, label) in dataset if label == 1 and len(text['sentence']) >= 3]
        dataset_class_1_t = dataset_class_1[:per_class_samples]
        incontext_dataset_class_1 = dataset_class_1[-5:]


        dataset_class_0 = [(text['sentence'], label) for (text, label) in dataset if label == 0 and len(text['sentence']) >= 3]
        dataset_class_0_t = dataset_class_0[:per_class_samples]
        incontext_dataset_class_0 = dataset_class_0[-5:]

        dataset_class = dataset_class_1_t + dataset_class_0_t
        print(f'Total filtered dataset size: {len(dataset_class)}')
        args.ceattack_logger.info(f"Total filtered dataset size: {len(dataset_class)}")
        return dataset_class, label_names

    elif task == 'ag_news':
        from textattack.datasets import HuggingFaceDataset
        from datasets import load_dataset  
        args.ceattack_logger.info(f"Loading {args.task} dataset from {dataset_location}")
        print(f"Loading {args.task} dataset from {dataset_location} ...")
        strategy_dataset = load_dataset(dataset_location, split='test')#.select(range(500)) # example: taking 50 samples
        print(f"{args.task} dataset loaded successfully", type(strategy_dataset), 'column names', strategy_dataset.column_names)
        args.ceattack_logger.info(f"{args.task} dataset loaded successfully {type(strategy_dataset)} column {strategy_dataset.column_names}")
        dataset = HuggingFaceDataset(strategy_dataset, split="test", shuffle=True)
        
        
        label_names = dataset.label_names
        per_class_samples = args.num_examples//args.n_classes

        
        dataset_class_0 = [(text['text'], label) for (text, label) in dataset if label == 0 and len(text['text']) >= 3]
        dataset_class_0_t = dataset_class_0[:per_class_samples]
        incontext_dataset_class_0 = dataset_class_0[-5:]

        
        dataset_class_1 = [(text['text'], label) for (text, label) in dataset if label == 1 and len(text['text']) >= 3]
        dataset_class_1_t = dataset_class_1[:per_class_samples]
        incontext_dataset_class_1 = dataset_class_1[-5:] 
        
        dataset_class_2 = [(text['text'], label) for (text, label) in dataset if label == 2 and len(text['text']) >= 3]
        dataset_class_2_t = dataset_class_2[:per_class_samples]
        incontext_dataset_class_2 = dataset_class_2[-5:]

        dataset_class_3 = [(text['text'], label) for (text, label) in dataset if label == 3 and len(text['text']) >= 3]
        dataset_class_3_t = dataset_class_3[:per_class_samples]
        incontext_dataset_class_3 = dataset_class_3[-5:]
        
        dataset_class =  dataset_class_1_t + dataset_class_0_t + dataset_class_2_t + dataset_class_3_t
        
        print(f'Total filtered dataset size: {len(dataset_class)}')
        args.ceattack_logger.info(f"Total filtered dataset size: {len(dataset_class)}")
        
        return dataset_class, label_names 

    elif task == 'popQA':
        from textattack.datasets import HuggingFaceDataset
        from datasets import load_dataset
        import json
        from collections import defaultdict
        
        
        split = "test"
        print(f"Loading {args.task} {split} dataset from {dataset_location}...")
        args.ceattack_logger.info(f"Loading {args.task} {split} dataset from {dataset_location}...")
        raw_dataset = load_dataset(dataset_location, split=split).select(range(5))
        print(f"{args.task} {split} dataset loaded successfully", type(raw_dataset),'column names',raw_dataset.column_names)
        args.ceattack_logger.info(f"{args.task} dataset loaded successfully {type(strategy_dataset)} column {strategy_dataset.column_names}")
        raise ValueError(f"The current version dosn't work with {args.task} but I kept the functionality in case someone wants to extend this." )
        
        

        prop_to_answers = {}
        # Populate the dictionary
        for sample in raw_dataset:
            
            if sample['prop'] in prop_to_answers:
                prop_to_answers[sample['prop']] |= set(json.loads(sample['possible_answers']))
            else:
                prop_to_answers[sample['prop']] = set(json.loads(sample['possible_answers']))
                
 


        wrong_possible_answers_col = []
        possible_correct_incorrect_answers_col = []
        for sample in raw_dataset:
            current_prop = sample['prop']
            
            current_answers = set(json.loads(sample['possible_answers']))
            
            
            all_answers_for_prop =  prop_to_answers[current_prop]
            
            wrong_answers = all_answers_for_prop - current_answers
            
            wrong_possible_answers_col.append(json.dumps(list(wrong_answers))) 

            possible_correct_incorrect_answers_col.append(json.dumps({
                'correct': list(current_answers),
                'incorrect': list(wrong_answers)
            }))
            
        raw_dataset = raw_dataset.add_column("wrong_possible_answers", wrong_possible_answers_col)
        raw_dataset = raw_dataset.add_column("possible_correct_incorrect_answers", possible_correct_incorrect_answers_col)

        
        
        
        raw_dataset = raw_dataset.rename_column("possible_correct_incorrect_answers", "label")
        raw_dataset = raw_dataset.rename_column("question", "text") 
        
        dataset = HuggingFaceDataset(raw_dataset,split="test", shuffle=True)
        print(f"Converted to PyTorch Dataset", type(dataset))
        label_names = None
 
 
        dataset_class = [] 
        dataset_class = [(text['text'], label) for (text, label) in dataset]
        dataset_class_t = dataset_class[:args.num_examples]
        incontext_dataset_class = dataset_class[-5:]

        print(f'Total filtered dataset size for PopQA: {len(dataset_class)}')
        print(f'In-context samples for PopQA: {incontext_dataset_class}')
        
        
        return dataset_class_t, label_names
    elif task == 'strategyQA':
        from textattack.datasets import HuggingFaceDataset
        from datasets import load_dataset  
        print(f"Loading {args.task} dataset from {dataset_location}...")
        args.ceattack_logger.info(f"Loading {args.task} dataset from {dataset_location}...")
        strategy_dataset = load_dataset(dataset_location, split='test').select(range(500)) # example: taking 50 samples
        print(f"{args.task} dataset loaded successfully", type(strategy_dataset), 'column names', strategy_dataset.column_names)
        args.ceattack_logger.info(f"{args.task} dataset loaded successfully {type(strategy_dataset)} column {strategy_dataset.column_names}")
        strategy_dataset = strategy_dataset.rename_column("question", "text")
        strategy_dataset = strategy_dataset.rename_column("answer", "label")

        strategy_dataset = HuggingFaceDataset(strategy_dataset, split="test", shuffle=True) 

        strategy_dataset_class = []
        strategy_dataset_class = [(text['text'], int(label)) for (text, label) in strategy_dataset]
        
        strategy_dataset_class_t = strategy_dataset_class[:args.num_examples] 
        strategy_incontext_dataset_class = strategy_dataset_class[-5:]
        label_names = ['false','true']
        
        return strategy_dataset_class_t, label_names
    elif task == 'triviaQA':
        from textattack.datasets import HuggingFaceDataset
        from datasets import load_dataset
        dataset_location = DATA_LOCATION[args.task] 
        split = 'validation'
        print(f"Loading TriviaQA {split} dataset from {dataset_location}...")
        args.ceattack_logger.info(f"Loading {args.task} {split} dataset from {dataset_location}...")
        dataset = load_dataset(dataset_location, "rc", split=split)
        print(f"{args.task} {split} dataset loaded successfully", len(dataset))
        args.ceattack_logger.info(f"{args.task} dataset loaded successfully {type(strategy_dataset)} column {strategy_dataset.column_names}")
        
        raise ValueError(f"The current version dosn't work with {args.task} but I kept the functionality case someone wants to extend this." )
        dataset = dataset.shuffle(seed=42).select(range(2000))
        dataset = dataset.filter(lambda example: example['entity_pages']['wiki_context'])
        label_names = ['false','true'] 
        
        dataset = dataset.rename_column("question_id", "title")
        def extract_context_and_rename(example):
            
            context = example['entity_pages']['wiki_context']
            context = ' '.join( context[0].split(' ')[:100])
            example['context'] = context
            return example

        def extract_answers_and_rename(example):
            
            example['answers'] = example['answer']['normalized_aliases'] 
            return example

        dataset = dataset.map(extract_context_and_rename)
        dataset = dataset.map(extract_answers_and_rename)

        dataset = dataset.remove_columns(['question_source'])
        
        dataset = dataset.remove_columns(['search_results'])
        dataset = dataset.remove_columns(['entity_pages'])

        dataset = HuggingFaceDataset(dataset, split="validation", shuffle=True)
        print(f"Converted triviaQA to PyTorch Dataset", type(dataset)) 
            
        dataset_class = []
        dataset_class = [((text['question'],text['context']), answers) for (text, answers) in dataset] 
        dataset_class_t = dataset_class[:args.num_examples] 
        incontext_dataset_class = dataset_class[-5:]
        

        print(f'Total filtered dataset size for triviaQA: {len(dataset_class_t)}')
        print(f'In-context samples for triviaQA: {incontext_dataset_class}') 
        return dataset_class_t , label_names

    elif task == 'mnli':
        from textattack.datasets import HuggingFaceDataset
        from datasets import load_dataset 
        args.ceattack_logger.info(f"Loading {args.task} dataset from {dataset_location}")
        print(f"Loading {args.task} dataset from {dataset_location}...")
        
        strategy_dataset = load_dataset({dataset_location}, split='validation_matched')
        print(f"{args.task} dataset loaded successfully", type(strategy_dataset), 'column names', strategy_dataset.column_names)
        args.ceattack_logger.info(f"{args.task} dataset loaded successfully {type(strategy_dataset)} column {strategy_dataset.column_names}")
        raise ValueError(f"The current version dosn't work with {args.task} but I kept the functionality case someone wants to extend this." )
        
        per_class_samples = args.num_examples // args.n_classes
        dataset = HuggingFaceDataset(strategy_dataset, split="validation", shuffle=True)
        label_names = dataset.label_names


        dataset_class_0 = [((text['premise'], text['hypothesis']), label) for (text, label) in dataset if label == 0]  # Neutral
        dataset_class_1 = [((text['premise'], text['hypothesis']), label) for (text, label) in dataset if label == 1]  # Entailment
        dataset_class_2 = [((text['premise'], text['hypothesis']), label) for (text, label) in dataset if label == 2]  # Contradiction


        dataset_class_0_t = dataset_class_0[:per_class_samples]
        incontext_dataset_class_0 = dataset_class_0[-5:]

        dataset_class_1_t = dataset_class_1[:per_class_samples]
        incontext_dataset_class_1 = dataset_class_1[-5:]

        dataset_class_2_t = dataset_class_2[:per_class_samples]
        incontext_dataset_class_2 = dataset_class_2[-5:]


        dataset_class = dataset_class_0_t + dataset_class_1_t + dataset_class_2_t
        
        return dataset_class, label_names
    elif task == 'rte':
        print(f"Loading {args.task} dataset...")
        args.ceattack_logger.info(f"Loading {args.task} dataset from {dataset_location}")
        dataset, label_names = load_huggingface_dataset(dataset_location[0], dataset_location[1], args.num_examples // 2)
        print(f"{args.task} dataset loaded successfully", type(strategy_dataset), 'column names', strategy_dataset.column_names)
        args.ceattack_logger.info(f"{args.task} dataset loaded successfully {type(strategy_dataset)} column {strategy_dataset.column_names}")       
        dataset_class_0 = [((text['sentence1'], text['sentence2']), label) for (text, label) in dataset if label == 0]
        dataset_class_1 = [((text['sentence1'], text['sentence2']), label) for (text, label) in dataset if label == 1]
        dataset_class_0_t = dataset_class_0[:args.num_examples // 2]
        dataset_class_1_t = dataset_class_1[:args.num_examples // 2]
        dataset_class = dataset_class_0_t + dataset_class_1_t
        return dataset_class, label_names

    elif task == 'qnli':
        print(f"Loading {args.task} dataset...")
        args.ceattack_logger.info(f"Loading {args.task} dataset from {dataset_location}")
        dataset, label_names = load_huggingface_dataset(dataset_location[0], dataset_location[1], args.num_examples // 2)
        print(f"{args.task} dataset loaded successfully", type(strategy_dataset), 'column names', strategy_dataset.column_names)
        args.ceattack_logger.info(f"{args.task} dataset loaded successfully {type(strategy_dataset)} column {strategy_dataset.column_names}")
        raise ValueError(f"The current version dosn't work with {args.task} but I kept the functionality case someone wants to extend this." )
        dataset_class_0 = [((text['question'], text['sentence']), label) for (text, label) in dataset if label == 0]
        dataset_class_1 = [((text['question'], text['sentence']), label) for (text, label) in dataset if label == 1]
        dataset_class_0_t = dataset_class_0[:args.num_examples // 2]
        dataset_class_1_t = dataset_class_1[:args.num_examples // 2]
        dataset_class = dataset_class_0_t + dataset_class_1_t
        return dataset_class, label_names

    elif task == 'qqp':
        print(f"Loading {args.task} dataset...")
        args.ceattack_logger.info(f"Loading {args.task} dataset from {dataset_location}")
        dataset, label_names = load_huggingface_dataset(dataset_location[0], dataset_location[1], args.num_examples // 2)
        print(f"{args.task} dataset loaded successfully", type(strategy_dataset), 'column names', strategy_dataset.column_names)
        args.ceattack_logger.info(f"{args.task} dataset loaded successfully {type(strategy_dataset)} column {strategy_dataset.column_names}")
        dataset_class_0 = [((text['question1'], text['question2']), label) for (text, label) in dataset if label == 0]
        dataset_class_1 = [((text['question1'], text['question2']), label) for (text, label) in dataset if label == 1]
        dataset_class_0_t = dataset_class_0[:args.num_examples // 2]
        dataset_class_1_t = dataset_class_1[:args.num_examples // 2]
        dataset_class = dataset_class_0_t + dataset_class_1_t
        return dataset_class, label_names


    else:
        print("Task not supported.")
        raise ValueError(f"The task {args.task} is not supported" )
        

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