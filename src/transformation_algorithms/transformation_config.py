
from textattack.transformations import WordSwapEmbedding

class HigherWordSwapEmbedding(WordSwapEmbedding):
    def __init__(self, **kwargs):
        # Initialize WordSwapEmbedding with max_candidates derived from n_embeddings in kwargs
        super().__init__(max_candidates=kwargs.get('n_embeddings'))

        # Store all additional kwargs as attributes of the instance
        for key, value in kwargs.items():
            setattr(self, key, value)

from .self_word_substitutions import SelfWordSubstitutionW1

DYNAMIC_TRANSFORMATION = {
    'sspattack':HigherWordSwapEmbedding,
    'texthoaxer':HigherWordSwapEmbedding,
    'ceattack':HigherWordSwapEmbedding,
    'self_word_sub':SelfWordSubstitutionW1,
}