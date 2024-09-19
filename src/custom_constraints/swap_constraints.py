import nltk
from textattack.constraints import Constraint
from textattack.shared.utils import LazyLoader
from textattack.shared.validators import transformation_consists_of_word_swaps
from textattack.transformations import WordSwap

nltk.download('averaged_perceptron_tagger')

class NoNounConstraint(Constraint):
    """A custom constraint to block any transformations of nouns."""

    def __init__(self, tagger_type="nltk", tagset="universal", language="eng"):
        super().__init__(compare_against_original=True)
        self.tagger_type = tagger_type
        self.tagset = tagset
        self.language = language

    def _get_pos(self, text):
        words = text.words
        pos_tags = nltk.pos_tag(words, tagset=self.tagset)
        return [pos for _, pos in pos_tags]

    def _check_constraint(self, transformed_text, reference_text):
        indices = transformed_text.attack_attrs["newly_modified_indices"]
        original_pos = self._get_pos(reference_text)

        for idx in indices:
            if original_pos[idx].startswith('N'):
                return False

        return True

    def check_compatibility(self, transformation):
        return transformation_consists_of_word_swaps(transformation)

