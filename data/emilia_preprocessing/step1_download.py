# conda activate emilia
from datasets import load_dataset
import fire


def main(root: str = "/data/scratch/pyp/datasets/emilia"):
    path = "EN/*.tar.gz*"
    dataset = load_dataset(
        "amphion/Emilia-Dataset",
        data_files={"en": path},
        split="en",
        streaming=False,
        revision="fc71e07e8572f5f3be1dbd02ed3172a4d298f152",
        cache_dir=root,
    )


if __name__ == "__main__":
    fire.Fire(main)
