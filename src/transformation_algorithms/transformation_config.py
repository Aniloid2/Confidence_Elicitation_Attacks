
from textattack.transformations import WordSwapEmbedding

class HigherWordSwapEmbedding(WordSwapEmbedding):
    def __init__(self, **kwargs):
        super().__init__(max_candidates=kwargs.get('n_embeddings'))

        for key, value in kwargs.items():
            setattr(self, key, value)

from .self_word_substitutions import SelfWordSubstitutionW1

DYNAMIC_TRANSFORMATION = {
    'sspattack':HigherWordSwapEmbedding,
    'texthoaxer':HigherWordSwapEmbedding,
    'ceattack':HigherWordSwapEmbedding,
    'self_word_sub':SelfWordSubstitutionW1,
}