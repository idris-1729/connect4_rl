from dataclasses import dataclass
from typing import Tuple
@dataclass
class Piece:
    turn : int
    coords : Tuple[int,int]