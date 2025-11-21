from typing import (
    Any
)

def filter(obj: Any, options: Any, parent: Any=None):
    if obj is None:
        return False
    if options is None:
        return True
    
    def _match(obj, key):
        if key==obj:
            return True
        if isinstance(key, str):
            if hasattr(obj, 'name'):
                name = options.get('name', None)
                if name is not None and name==key:
                    return True
        if isinstance(key, type):
            if isinstance(objy, key):
                return True            
        return False            
    
    include = options.get('include', None)
    if include is not None:
        if isinstance(include, str):
            include = [ s.strip() for s in include.split(',') ]        
        if isinstance(include, list):
            for key in include:
                if _match(obj, key):
                    return True
        return False
    exclude = options.get('exclude', None)
    if exclude is not None:
        if isinstance(exclude, str):
            exclude = [ s.strip() for s in exclude.split(',') ]        
        if isinstance(exclude, list):
            for key in include:
                if _match(obj, key):
                    return False
        else:
            if _match(obj, exclude):
                return False
            
    return True

def filter_count(objs: Any, options: Any, parent: Any=None):
    n = 0
    for obj in objs:
        if filter(obj, options, parent):
            n += 1
    return n
    