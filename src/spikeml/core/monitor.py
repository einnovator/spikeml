from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mplt

class Monitor:
    """
    A simple monitor class to track and store properties of a reference object over time.

    Attributes
    ----------
    name : Optional[str]
        Name of this monitor instance.
    ref : Optional[Any]
        Reference object whose properties are being monitored.
    """

    def __init__(self, name: Optional[str] = None, ref: Optional[Any] = None) -> None:
        """
        Initialize a Monitor instance.

        Parameters
        ----------
        name : str, optional
            Name of the monitor.
        ref : object, optional
            Reference object to monitor.
        """
        #super().__init__()
        self.name = name
        self.ref = ref

    def _sample(self, key: str, value: Any) -> "Monitor":
        """
        Append a value to the monitored list corresponding to `key`.

        If the attribute for the key does not exist or is None, it initializes
        it as an empty list before appending.

        Parameters
        ----------
        key : str
            Name of the property to sample.
        value : Any
            Value to append to the monitored list.

        Returns
        -------
        Monitor
            Returns self for method chaining.
        """
        if value is not None:
            if not hasattr(self, key) or getattr(self, key) is None:
                setattr(self, key, self._make_empty())
            getattr(self, key).append(value)
        return self

    def _make_empty(self):
        """
        Create an empty list for storing samples.

        Returns
        -------
        list
            An empty list.
        """
        return []
    
    def _get(self, prop: str, ref: Optional[Any] = None, copy: bool = True) -> Any:
        """
        Retrieve the value of a property from the reference object.

        Parameters
        ----------
        prop : str
            Name of the property to retrieve.
        ref : object, optional
            Reference object. If None, uses self.ref.
        copy : bool, default True
            Whether to return a copy of the value (for mutable types like numpy arrays).

        Returns
        -------
        Any
            Value of the property, or None if it does not exist.
        """
        
        if ref is None:
            ref = self.ref        
        if hasattr(ref, prop):
            value = getattr(ref, prop)
            if value is not None and copy:
                value = np.copy(value)
            return value
        else:
            return None
        
    def _sample_prop(self, key: str, prop: Optional[str] = None, ref: Optional[Any] = None, copy: bool = True) -> None:
        """
        Sample a property from the reference object and store it in the monitor.

        Parameters
        ----------
        key : str
            Attribute name in the monitor where the value will be stored.
        prop : str, optional
            Name of the property to sample from the reference. Defaults to `key`.
        ref : object, optional
            Reference object to sample from. Defaults to self.ref.
        copy : bool, default True
            Whether to copy the value (important for numpy arrays or mutable objects).
        """
        if ref is None:
            ref = self.ref
        if prop is None:
            prop = key
        self._sample(key, self._get(prop, ref=ref, copy=copy))
        
    def _prefix(self, prefix: Optional[str] = None) -> str:
        """
        Determine the prefix string for the monitor or reference name.

        Priority:
        1. Self name if defined.
        2. Reference object's name if available.
        3. Empty string if neither exists.

        Parameters
        ----------
        prefix : str, optional
            Initial prefix to override. Defaults to None.

        Returns
        -------
        str
            Resolved prefix string.
        """
        if self.name is not None and len(self.name)> 0:
            prefix = f'{self.name}'
        elif hasattr(self, 'ref'):
            ref = getattr(self, 'ref')
            if ref is not None:
                name = ref.name
                if name is not None and len(name)> 0:
                    prefix = f'{name}'
        if prefix is None:
            prefix = ''
        return prefix
