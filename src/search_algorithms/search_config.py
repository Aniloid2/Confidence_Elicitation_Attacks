from .sspattack_search import SSPAttackSearch
from .texthoaxer_search import TextHoaxerSearch
from .greedy_use_search import GreedyUSESearch
from .black_box_search import BlackBoxSearch

DYNAMIC_SEARCH = {
    

        'sspattack_search': SSPAttackSearch, 
        'black_box_search': BlackBoxSearch,
        'greedy_use_search':GreedyUSESearch,
        'texthoaxer_search':TextHoaxerSearch,

    }