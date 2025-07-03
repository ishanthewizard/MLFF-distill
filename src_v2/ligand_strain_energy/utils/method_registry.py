METHODS_DICT = {}
METHODS_DICT_ENERGY = {}

def register_method(name, func, category="minimisation"):
    if category == "minimisation":
        METHODS_DICT[name] = func
    elif category == "energy":
        METHODS_DICT_ENERGY[name] = func
    else:
        raise ValueError(f"Unknown category: {category}")

def get_method(name, category="minimisation"):
    if category == "minimisation":
        return METHODS_DICT[name]
    elif category == "energy":
        return METHODS_DICT_ENERGY[name]
    else:
        raise ValueError(f"Unknown category: {category}")

def get_all_methods(category="minimisation"):
    if category == "minimisation":
        return list(METHODS_DICT.keys())
    elif category == "energy":
        return list(METHODS_DICT_ENERGY.keys())
    else:
        raise ValueError(f"Unknown category: {category}")