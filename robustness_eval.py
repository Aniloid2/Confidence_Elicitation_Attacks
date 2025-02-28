 
from transformers import  AutoTokenizer, AutoModelForCausalLM 
 

from src.arg_parser.arg_config import get_args
args = get_args()  
from src.arg_parser.set_cache import set_huggingface_cache
set_huggingface_cache(args)


 
from src.utils.shared.misc import environment_setup
args = environment_setup(args) 
 
 
from src.utils.shared.globals import CONFIDENCE_LEVELS, CONFIDENCE_MAP,TASK_N_CLASSES,MODEL_INFO


 


args.n_classes =  TASK_N_CLASSES[args.task] 
args.confidence_type_dict = CONFIDENCE_LEVELS[args.confidence_type] 
args.confidence_map_dict = CONFIDENCE_MAP[args.confidence_type]  
model_info = MODEL_INFO[args.model_type]



from src.utils.shared import load_data
data_to_evaluate, label_names = load_data(args)

 

from src.utils.shared import SimpleDataset
# Convert filtered datasets into TextAttack Dataset format 
args.dataset =  SimpleDataset(data_to_evaluate,label_names = label_names ) 



print("Dataset loaded successfully.",args.dataset) 


args.model_name =  model_info['model_name']
args.start_prompt_header = model_info['start_prompt_header']
args.end_prompt_footer = model_info['end_prompt_footer']



from src.utils.shared.misc import initialize_device
args.device = initialize_device(args)



from src.containers import AbstractPredictor, BasePredictorResults, ClassifierPredictorResults

class PredictionContainer(AbstractPredictor):
    def __init__(self):
        self.base_results = BasePredictorResults()
        self.classifier_results = ClassifierPredictorResults()



from src.llm_wrappers.huggingface_llm_wrapper import HuggingFaceLLMWrapper
from src.llm_wrappers.chatgpt_llm_wrapper import ChatGPTLLMWrapper

if 'gpt-4o' in args.model_type: 
    model_wrapper = ChatGPTLLMWrapper(**vars(args))
else:
    
    args.tokenizer = AutoTokenizer.from_pretrained(args.model_name ,cache_dir=args.cache_transformers,trust_remote_code=True  )
    args.model = AutoModelForCausalLM.from_pretrained(args.model_name , cache_dir=args.cache_transformers,trust_remote_code=True)
    args.model.to(args.device)
    model_wrapper = HuggingFaceLLMWrapper(**vars(args)) 
args.model = model_wrapper





from src.inference.inference_config import DYNAMIC_INFERENCE
args.predictor = DYNAMIC_INFERENCE[args.prompting_type](**vars(args))
args.predictor.predictor_container = PredictionContainer()


for datapoint in args.dataset:
    
    text, true_label = datapoint 
    
    guess, probs, confidence = args.predictor.predict_and_confidence(text)
    
        
    prediction_label = args.predictor.prompt_class.task_name_to_label[guess] #TASK_NAME_TO_LABEL

    
    args.predictor.predictor_container.add_true_label(true_label)
    args.predictor.predictor_container.add_probability(probs)
    args.predictor.predictor_container.add_confidence(confidence)
    
 


from src.utils.shared.evaluation.predictor_evaluation import predictor_evaluation

predictor_evaluation(args,args.predictor)




