import os
from .examples import NMTExample
import pandas as pd
from tqdm import tqdm

DATAPATH = {"train": "train.csv", "dev": "dev.csv", "test": "test_official.csv", "test2": "test2_official.csv"}

# train, dev, test, test2 반환
def load_dataset(args, source_tokenizer, target_tokenizer):
    cache_path = os.path.join(args.root, "cache")

    if not os.path.isdir(cache_path):
        os.makedirs(cache_path)

        train_pairs = get_pairs_from_dataset(args,"train")  # dic[list] 형태
        train_examples = convert_data_to_examples(source_tokenizer, target_tokenizer, train_pairs)
        pd.to_pickle(train_examples, os.path.join(cache_path, "train.pkl"))

        dev_pairs = get_pairs_from_dataset(args,"dev")
        dev_examples = convert_data_to_examples(source_tokenizer, target_tokenizer, dev_pairs)
        pd.to_pickle(dev_examples, os.path.join(cache_path, "dev.pkl"))

        test_pairs = get_pairs_from_dataset(args, "test", test=True)
        test_examples = convert_data_to_examples(source_tokenizer, target_tokenizer, test_pairs, test=True)
        pd.to_pickle(test_examples, os.path.join(cache_path, "test.pkl"))

        test2_pairs = get_pairs_from_dataset(args, "test2", test=True)
        test2_examples = convert_data_to_examples(source_tokenizer, target_tokenizer, test2_pairs, test=True)
        pd.to_pickle(test2_examples, os.path.join(cache_path, "test2.pkl"))

    else:
        train_examples = pd.read_pickle(os.path.join(cache_path, "train.pkl"))
        dev_examples = pd.read_pickle(os.path.join(cache_path, "dev.pkl"))
        test_examples = pd.read_pickle(os.path.join(cache_path, "test.pkl"))
        test2_examples = pd.read_pickle(os.path.join(cache_path, "test2.pkl"))


    return train_examples, dev_examples, test_examples, test2_examples

# train/ test -> sid, en, ko/ sid, en 가져와 dic 로 반환
def get_pairs_from_dataset(args, data_type, test=False):
    df = pd.read_csv(os.path.join(args.root, DATAPATH[data_type]))
    sid = df["sid"].to_list()
    src_language = df["en"].to_list()
    if test:
        return {"sid": sid, "src": src_language}
    else:
        trg_language = df["ko"].to_list()
        return {"sid": sid, "src": src_language, "trg": trg_language}

# model 에 넣기위한? (guid, input_ids, trg_ids) 로 다룰 수 있게 클래스화 시킨다 <list[class]>
# NMTExample() ->
def convert_data_to_examples(source_tokenizer, target_tokenizer, dataset, test=False):
    examples = []

    if test:
        for idx, (sid, src) in tqdm(enumerate(zip(dataset["sid"], dataset["src"]))):
            src_ids = source_tokenizer.encode(src.strip())
            examples.append(NMTExample(guid=f"{sid}", input_ids=src_ids,trg_ids=None))
    else:
        for idx, (sid, src, trg) in tqdm(enumerate(zip(dataset["sid"], dataset["src"], dataset["trg"]))):
            src_ids = source_tokenizer.encode(src.strip())
            trg_ids = target_tokenizer.encode(trg.strip())
            examples.append(NMTExample(guid=f"{sid}", input_ids=src_ids, trg_ids=trg_ids))

    return examples
