
from msc.constants import *
import torch



def label_converter_pyfeat(curr_aus_scores, all_aus):
    """
    Converts the input label based on the provided mapping.

    Args:
        input_label (str): The label to be converted.
        label_mapping (dict): A dictionary mapping original labels to new labels.

    Returns:
        str: The converted label if found in the mapping, otherwise returns the original label.
    """
    has = [False] * len(all_aus)
    intensity = [0] * len(all_aus)
    binary = [False] * len(all_aus)
    for au in SUPPORTED_AUS_PYFEAT:
        if au in all_aus:
            idx = all_aus.index(au)
            has[idx] = True
            binary[idx] = False
            intensity[idx] = curr_aus_scores[au]
    return has, binary, intensity


def label_stanardizer(input_label, label_mapping, type_="pyfeat"):
    """
    Standardizes the input label based on the provided mapping.

    Args:
        input_label (str): The label to be standardized.
        label_mapping (dict): A dictionary mapping original labels to standardized labels.

    Returns:
        str: The standardized label if found in the mapping, otherwise returns the original label.
    """
    type_ = type_.lower()
    all_aus = AU_TO_FACS_MAP.keys()
    au_count = len(AU_TO_FACS_MAP)
    output_tensor = torch.zeros(output_dims)
    if type_ == "pyfeat":
        convert_dict = SUPPORTED_AUS_PYFEAT
        # map the AU lables of the SUPPORTED_AUS_PYFEAT to the AU_TO_FACS_MAP and set the corresponding index in the output tensor to 1 if the input label is in the convert_dict
        
        has, intensity, binary = label_converter_pyfeat(input_label, all_aus, convert_dict)

    # (3, au_count), first dimension is True False, second is Intensity, third is true false depending on if method has given AU or not
    output_dims = (3, au_count)
    torch_output = torch.zeros(output_dims)
    torch_output[0] = torch.tensor(has)
    torch_output[1] = torch.tensor(intensity)
    torch_output[2] = torch.tensor(binary)

    return torch_output


if __name__ == "__main__":
    # Example usage
    label_mapping = {
        "AU01": "Inner Brow Raiser",
        "AU02": "Outer Brow Raiser",
        # Add more mappings as needed
    }

    input_label = "AU01"
    standardized_label = label_stanardizer(input_label, label_mapping)
    print(f"Standardized Label: {standardized_label}")