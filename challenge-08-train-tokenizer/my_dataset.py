# "low-res" copy of https://github.com/karpathy/nanochat/blob/master/nanochat/dataset.py

import requests
import os
import pyarrow.parquet as pq

BASE_URL = "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
index_to_filename = lambda index: f"shard_{index:05d}.parquet" # format of the filenames

def download_single_file(index):
    filename = index_to_filename(index)
    url = f"{BASE_URL}/{filename}"
    print(f"downloading {filename}...")

    max_attempts = 5
    for attempt in range(1, max_attempts+1):
        try:
            response = requests.get(url, stream = True, timeout = 30)
            response.raise_for_status()
            temp_path = f"{filename}.tmp"
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size = 1024 * 1024):
                    if chunk:
                        f.write(chunk)
            os.rename(temp_path, filename)
            print(f"downloaded {filename}")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt} to download {filename} failed: {e}.")
            for path in [f"{filename}.tmp", filename]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except:
                        print(f"failed to remove {path}");
            if attempt < max_attempts:
                wait_time = 2 ** attempt
                print(f"will wait {wait_time} seconds before retrying download of {filename}")
                time.sleep(wait_time)
            else:
                print(f"failed to download {filename} after multiple attempts")
                return False

def list_parquet_files():
    data_dir = '.'
    return sorted([
        f for f in os.listdir(data_dir)
        if f.endswith('parquet') and not f.endswith('.tmp')
    ])

# his notes say start / step is to support DDP so I imagine later we'll see giving
# n parallel processes different starts with step=n, but his notes mention step=world_size
# and not sure what that means, sure will get to it later
def parquets_iter_batched(split, start=0, step=1):
    assert split in ["train", "val"]
    parquet_paths = list_parquet_files()
    parquet_paths = parquet_paths[:-1] if split == 'train' else parquet_paths[-1:]
    for filepath in parquet_paths:
        pf = pq.ParquetFile(filepath)
        # claude: A row group is a horizontal partition of data in a Parquet file.
        # It's a logical grouping of rows that are stored together.
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column('text').to_pylist()
            yield texts


# "signature copied" from https://github.com/karpathy/nanochat/blob/master/scripts/tok_train.py
# this is the iterator we're going to need for training the tokenizer
# his doc:
#    1) Flatten the batches into a single iterator
#    2) Crop every document to args.doc_cap characters
#    3) Break when we've seen args.max_chars characters
# 
def text_iterator(max_chars=10_000_000_000, doc_cap=10_000):
    total_chars = 0
    for texts in parquets_iter_batched('train'):
        for doc in texts:
            doc = doc[:min((max_chars - total_chars),doc_cap)]
            yield doc
            total_chars += len(doc)
            if (total_chars >= max_chars):
                return

