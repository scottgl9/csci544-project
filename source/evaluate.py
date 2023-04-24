from metrics.stereoset import evaluate_stereoset_intrasentence
from metrics.crows_pairs import evaluate_crows_pairs
from utils import convert_to_hf_model
from config import DEVICE


def evaluate(ckpt_path, dataset, **kwargs):
    model = convert_to_hf_model(ckpt_path)
    model = model.to(DEVICE)
    if dataset == "stereoset":
        evaluate_stereoset_intrasentence(model, device=DEVICE)
    elif dataset == "crows":
        evaluate_crows_pairs(model, device=DEVICE)
    else:
        raise NotImplementedError("dataset should be either streoset or crows")
    return
