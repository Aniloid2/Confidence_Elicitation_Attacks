

from .classification import *

DYNAMIC_PROMPT = {
    

        'classification': 
            {
                'ag_news':AgNewsPrompts,
                'strategyQA':StrategyQAPrompts,
                'sst2':SST2Prompts,
                'mnli':BaseClassificationPrompt,
            },


        'sequence_to_sequence':
            {
                'to_implement':AgNewsPrompts
            },


    }