# Adapted from  https://github.com/facebookresearch/esm/blob/main/scripts/extract.py

import pathlib

import torch
from esm import (  # Alphabet,; ProteinBertModel,
    FastaBatchedDataset,
    MSATransformer,
    pretrained,
)


def generate_esm_embeddings(
    model_location="",
    fasta_file: pathlib.Path = "",
    output_dir: pathlib.Path = "",
    include="per_tok",
    use_gpu=True,
):
    fasta_file = pathlib.Path(fasta_file)
    output_dir = pathlib.Path(output_dir)

    truncation_seq_length = 1022  # default in the original code.
    toks_per_batch = 4096  # default in the original code.
    repr_layers = [-1]

    print("Load pretrained ESM model and alphabet")
    if model_location.startswith(".") or model_location.startswith("/"):
        model, alphabet = pretrained.load_model_and_alphabet_local(model_location)
    else:
        model, alphabet = pretrained.load_model_and_alphabet(model_location)
    print("... done!")

    model.eval()
    if isinstance(model, MSATransformer):
        raise ValueError(
            "This script currently does not handle models with MSA input (MSA Transformer)."
        )

    if torch.cuda.is_available() and use_gpu:
        model = model.to("cuda")
        print("Transferred model to GPU")

    dataset = FastaBatchedDataset.from_file(fasta_file)
    batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=alphabet.get_batch_converter(truncation_seq_length),
        batch_sampler=batches,
    )
    print(f"Read {fasta_file} with {len(dataset)} sequences")

    output_dir.mkdir(parents=True, exist_ok=True)
    return_contacts = "contacts" in include

    assert all(-(model.num_layers + 1) <= i <= model.num_layers for i in repr_layers)
    repr_layers = [
        (i + model.num_layers + 1) % (model.num_layers + 1) for i in repr_layers
    ]

    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(
                f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)"
            )
            if torch.cuda.is_available() and use_gpu:
                toks = toks.to("cuda", non_blocking=True)

            out = model(toks, repr_layers=repr_layers, return_contacts=return_contacts)

            # logits = out["logits"].to(device="cpu")
            representations = {
                layer: t.to("cpu") for layer, t in out["representations"].items()
            }
            if return_contacts:
                contacts = out["contacts"].to(device="cpu")

            for i, label in enumerate(labels):
                output_file = output_dir / f"{label}.pt"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                result = {"label": label}
                truncate_len = min(truncation_seq_length, len(strs[i]))
                # Call clone on tensors to ensure tensors are not views into a larger representation
                # See https://github.com/pytorch/pytorch/issues/1995
                if "per_tok" in include:
                    result["representations"] = {
                        layer: t[i, 1 : truncate_len + 1].clone()
                        for layer, t in representations.items()
                    }

                if "mean" in include:
                    result["mean_representations"] = {
                        layer: t[i, 1 : truncate_len + 1].mean(0).clone()
                        for layer, t in representations.items()
                    }
                if "bos" in include:
                    result["bos_representations"] = {
                        layer: t[i, 0].clone() for layer, t in representations.items()
                    }
                if return_contacts:
                    result["contacts"] = contacts[
                        i, :truncate_len, :truncate_len
                    ].clone()

                torch.save(
                    result,
                    output_file,
                )
