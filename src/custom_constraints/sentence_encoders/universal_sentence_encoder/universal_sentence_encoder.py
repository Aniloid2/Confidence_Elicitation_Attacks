"""
universal sentence encoder class
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""

from textattack.constraints.semantics.sentence_encoders import SentenceEncoder
from textattack.shared.utils import LazyLoader
import torch
import math

hub = LazyLoader("tensorflow_hub", globals(), "tensorflow_hub")


class UniversalSentenceEncoder(SentenceEncoder):
    """Constraint using similarity between sentence encodings of x and x_adv
    where the text embeddings are created using the Universal Sentence
    Encoder."""

    def __init__(self, threshold=0.8, large=False, metric="angular", **kwargs):
        super().__init__(threshold=threshold, metric=metric, **kwargs)
        if large:
            tfhub_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
        else:
            tfhub_url = "https://tfhub.dev/google/universal-sentence-encoder/3"

        self._tfhub_url = tfhub_url
        # Lazily load the model
        self.model = None

    def encode(self, sentences):
        if not self.model:
            
            self.model = hub.load(self._tfhub_url)

        encoding = self.model(sentences)
        
        if isinstance(encoding, dict):
            encoding = encoding["outputs"]

        return encoding.numpy()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["model"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.model = None

    def get_angular_sim(emb1, emb2):
        """Returns the _angular_ similarity between a batch of vector and a batch
        of vectors.""" 
        cos_sim = torch.nn.CosineSimilarity(dim=1)(emb1, emb2)
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0) 
        return 1 - (torch.acos(cos_sim) / math.pi)

    def get_sim_score(self, text1,text2):
        sim_final_original, sim_final_pert = self.encode([text1, text2])

        if not isinstance(sim_final_original, torch.Tensor):
            sim_final_original = torch.tensor(sim_final_original)

        if not isinstance(sim_final_pert, torch.Tensor):
            sim_final_pert = torch.tensor(sim_final_pert)

        sim_score = self.sim_metric(sim_final_original.unsqueeze(0), sim_final_pert.unsqueeze(0)).item()
        sim_score = round(sim_score, 4)
        return sim_score
                
