
import os
def set_huggingface_cache(args):
    cache_dir = args.cache_transformers
    os.environ['HF_DATASETS_CACHE'] = cache_dir + 'datasets' 
    args.ceattack_logger.info(f"Set huggingface HF_DATASETS_CACHE at '{cache_dir}datasets'")