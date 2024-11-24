

from .classification import *
from .generation import *

DYNAMIC_PROMPT = {
    

        'classification': 
            {
                'ag_news':AgNewsPrompts,
                'strategyQA':StrategyQAPrompts,
                'sst2':SST2Prompts,
                'mnli':MNLIPrompts,
                'rte':RTEPrompts,
                'qnli':QNLIPrompts,
                'qqp':QQPPrompts,
            },
        'generation':
            {
                'triviaQA': TriviaQAPrompts
             },


        'sequence_to_sequence':
            {
                'to_implement':TriviaQAPrompts
            },


    }