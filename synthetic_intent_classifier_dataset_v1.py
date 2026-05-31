import datasets


class GuardrailDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            description="A simple binary guardrail dataset for classifying text as safe (0) or unsafe (1).",
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "label": datasets.ClassLabel(names=[
                        "greeting", "farewell", "thank_you", "affirmation", "negation", "small_talk", 
                        "bot_capabilities", "feedback_positive", "feedback_negative", "clarification", 
                        "suggestion", "language_change"
                    ]),
                }
            ),
            supervised_keys=("text", "label"),
            homepage="https://huggingface.co/datasets/tanaos/synthetic-intent-classifier-dataset-v1",
            license="mit",
        )

    def _split_generators(self, dl_manager):
        # The dataset only has one file
        data_path = self.config.data_dir or "./data/data.csv"
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": data_path},
            ),
        ]

    def _generate_examples(self, filepath):
        """
        Yields examples as (key, example) tuples.
        """
        
        import csv

        with open(filepath, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                yield i, {
                    "text": row["text"],
                    "label": int(row["label"]),
                }
