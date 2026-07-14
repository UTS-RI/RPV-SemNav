# Author: Adam Scicluna
# This script will load in a YAML file containing the LVIS class names and populate the LVIS_CLASSES list with those names.

import yaml

def load_lvis_class_names(yaml_path, include_all_names=False):
    """
    Load LVIS class names from a YAML file.
    Args:
        yaml_path (str): Path to the LVIS YAML file.
        include_all_names (bool): If True, include all names in each category (split by '/').
            If False, only the first name is used for each category.
    Returns:
        list[str]: List of class names.
    """
    with open(yaml_path, 'r') as f:
        lvis_data = yaml.safe_load(f)

    if include_all_names:
        # Flatten all names, split by '/', deduplicate (using a set; order not preserved)
        class_names = list({n.strip() for name in lvis_data['names'].values() for n in name.split('/') if n.strip()})
        return class_names
    else:
        class_names = [name.split('/')[0].strip() for name in lvis_data['names'].values()]
        return class_names