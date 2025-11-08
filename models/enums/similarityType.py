from pyparsing import Enum

class AbsSimilarityType(Enum):
    WITHIN = "within_abs"
    BETWEEN = "between_abs"
