
import os
def set_huggingface_cache(args):
    cache_dir = args.cache_transformers
    os.environ['HF_DATASETS_CACHE'] = cache_dir + 'datasets' 