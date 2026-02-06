import csv
import json
from typing import List, Tuple, Union, Optional
import pandas as pd
from pathlib import Path


class ConceptDataset:
    def __init__(
        self,
        path: Union[str, Path],
        *,
        prompt_field: str = "prompt",
        json_key: Optional[str] = None,
        dedup: bool = False,
    ):
        """
        Initialize the dataset by loading data from CSV / JSON / JSONL.

        Args:
            path: Path to a .csv, .json, or .jsonl file.
            prompt_field: Column/key name that holds the prompt string (default: 'prompt').
            json_key: If the top-level JSON is a dict and you only want a specific key's list,
                      provide its name. If None, all list-like values are concatenated.
            dedup: If True, remove duplicate prompts while preserving order.
        """
        self.path = Path(path)
        self.prompt_field = prompt_field
        self.json_key = json_key
        self.data: List[str] = []

        suffix = self.path.suffix.lower()
        if suffix == ".csv":
            self._load_csv()
        elif suffix == ".json":
            self._load_json()
        elif suffix == ".jsonl":
            self._load_jsonl()
        else:
            raise ValueError(f"Unsupported file type: {self.path.suffix} (use .csv, .json, or .jsonl)")

        if dedup:
            self._deduplicate_in_place()

    def _extract_prompt_from_dict(self, d: dict) -> Optional[str]:
        if not isinstance(d, dict):
            return None

        val = d.get(self.prompt_field)
        if isinstance(val, str) and val.strip():
            return val.strip()

        for k in ("sentence", "prompt", "text"):
            val = d.get(k)
            if isinstance(val, str) and val.strip():
                return val.strip()

        return None

    def _load_csv(self):
        with self.path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if self.prompt_field in row and isinstance(row[self.prompt_field], str):
                    p = row[self.prompt_field].strip()
                    if p:
                        self.data.append(p)

    def _load_json(self):
        with self.path.open("r", encoding="utf-8") as f:
            obj = json.load(f)

        if isinstance(obj, list):
            self._extend_from_sequence(obj)

        elif isinstance(obj, dict):
            # If user specified a particular key, read only that
            if self.json_key is not None:
                seq = obj.get(self.json_key, [])
                self._extend_from_sequence(seq)
            else:
                for seq in obj.values():
                    self._extend_from_sequence(seq)
        else:
            raise ValueError("Unsupported JSON structure: expected list or dict at the top level.")

    def _load_jsonl(self):
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("{") or line.startswith("["):
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    if isinstance(obj, dict):
                        p = self._extract_prompt_from_dict(obj)
                        if p:
                            self.data.append(p)
                    elif isinstance(obj, list):
                        self._extend_from_sequence(obj)
                else:
                    # Treat as raw string line
                    self.data.append(line)

    def _extend_from_sequence(self, seq):
        if not isinstance(seq, list):
            return
        for item in seq:
            if isinstance(item, str):
                p = item.strip()
                if p:
                    self.data.append(p)
            elif isinstance(item, dict):
                p = self._extract_prompt_from_dict(item)
                if p:
                    self.data.append(p)

    def _deduplicate_in_place(self):
        seen = set()
        deduped = []
        for p in self.data:
            if p not in seen:
                seen.add(p)
                deduped.append(p)
        self.data = deduped
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def get_batches(self, batch_size: int) -> List[dict]:
        """
        Group the data into batches of prompts.

        Args:
            batch_size: Number of samples per batch.

        Returns:
            List[dict]: A list of batches where each batch is {'prompt': List[str]}.
        """
        batches = []
        for i in range(0, len(self.data), batch_size):
            batch_data = self.data[i:i + batch_size]
            batches.append({'prompt': list(batch_data)})
        return batches


class SupervisedConceptDataset:
    def __init__(self, path: str):
        """
        Initialize the dataset by loading the data from a CSV or JSON file.

        Supported formats (in addition to the original ones):
          - List[dict] JSON with fields like:
              {'parent': ..., 'level': ..., 'concept': <LABEL>, 'sentence': <PROMPT>}
            -> label := concept, prompt := sentence
        """
        self.path = path
        self.data: List[Tuple[str, str]] = []

        def _add_pairs_from_df(df: pd.DataFrame, prompt_col: str, label_col: str):
            sub = df[[prompt_col, label_col]].dropna(subset=[prompt_col, label_col])
            for p, y in zip(sub[prompt_col], sub[label_col]):
                if isinstance(p, str) and isinstance(y, str):
                    p2, y2 = p.strip(), y.strip()
                    if p2 and y2:
                        self.data.append((p2, y2))
                else:
                    p2 = "" if p is None else str(p).strip()
                    y2 = "" if y is None else str(y).strip()
                    if p2 and y2:
                        self.data.append((p2, y2))

        if path.endswith(".csv"):
            df = pd.read_csv(self.path, encoding="utf-8")

            if {"prompt", "label"}.issubset(df.columns):
                _add_pairs_from_df(df, "prompt", "label")
            elif {"text", "label"}.issubset(df.columns):
                _add_pairs_from_df(df, "text", "label")
            elif {"sentence", "concept"}.issubset(df.columns):
                _add_pairs_from_df(df, "sentence", "concept")
            elif {"sentence", "label"}.issubset(df.columns):
                _add_pairs_from_df(df, "sentence", "label")

        elif path.endswith(".json"):
            df = None
            try:
                df = pd.read_json(self.path, encoding="utf-8")
            except ValueError:
                try:
                    df = pd.read_json(self.path, orient="index", encoding="utf-8")
                except ValueError:
                    df = None

            if df is not None and isinstance(df, pd.DataFrame):
                if {"prompt", "label"}.issubset(df.columns):
                    _add_pairs_from_df(df, "prompt", "label")
                elif {"text", "label"}.issubset(df.columns):
                    _add_pairs_from_df(df, "text", "label")
                # New format: sentence + concept (label := concept)
                elif {"sentence", "concept"}.issubset(df.columns):
                    _add_pairs_from_df(df, "sentence", "concept")
                elif {"sentence", "label"}.issubset(df.columns):
                    _add_pairs_from_df(df, "sentence", "label")
                else:
                    df = None

            if df is None:
                with open(self.path, "r", encoding="utf-8") as f:
                    loaded_data = json.load(f)

                if isinstance(loaded_data, list):
                    for item in loaded_data:
                        if not isinstance(item, dict):
                            continue
                        prompt = item.get("prompt") or item.get("text") or item.get("sentence")
                        label = item.get("label") or item.get("concept")
                        if prompt is None or label is None:
                            continue
                        p2, y2 = str(prompt).strip(), str(label).strip()
                        if p2 and y2:
                            self.data.append((p2, y2))

                elif isinstance(loaded_data, dict):
                    for label, prompts in loaded_data.items():
                        if label is None or prompts is None:
                            continue
                        for prompt in (prompts if isinstance(prompts, list) else []):
                            if prompt is None:
                                continue
                            p2, y2 = str(prompt).strip(), str(label).strip()
                            if p2 and y2:
                                self.data.append((p2, y2))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[str, str]:
        return self.data[idx]

    def get_batches(self, batch_size: int) -> List[dict]:
        batches = []
        for i in range(0, len(self.data), batch_size):
            batch = self.data[i:i + batch_size]
            prompts, labels = zip(*batch) if batch else ([], [])
            batches.append({"prompt": list(prompts), "label": list(labels)})
        return batches
