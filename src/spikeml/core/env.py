from typing import Annotated, Any, Callable, Optional, Sequence, Union, Tuple, Dict, List

class Source():
    """
    Base class for signal sources.

    Attributes:
        name: Optional name of the Signal.
    """
    def __init__(self,         
                 name: Optional[str] = None,
                 ):
        #super().__init__(name=name)
        self.name = name
        
        
class World(Source):
    """
    Base class for signal world.

    Attributes:
        name: Optional name of the World.
    """
    def __init__(self,         
                 name: Optional[str] = None,
                 ):
        pass
