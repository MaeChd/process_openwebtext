import os
import glob
import json
import numpy as np
import lzma
import tarfile
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset
import tiktoken


def decompress_xz_files(src_dir, dest_dir):
    """Decompress all .xz files from the source directory and copy them as .txt files to the destination directory."""
    os.makedirs(dest_dir, exist_ok=True)

    # Get all .xz files in the source directory
    xz_files = [f for f in os.listdir(src_dir) if f.endswith('.xz')]

    for idx, xz_file in enumerate(xz_files, 1):
        src_file = os.path.join(src_dir, xz_file)
        if os.path.exists(src_file):
            # Check if the file is a tarball
            if tarfile.is_tarfile(src_file):
                with tarfile.open(src_file, mode='r:xz') as tar:
                    tar.extractall(path=dest_dir)
                    print(f"Extracted all contents of {src_file} to {dest_dir}")
            else:
                dest_file_path = os.path.join(dest_dir, f"extracted_content_{idx}.txt")
                with lzma.open(src_file, 'rt') as file:
                    content = file.read()
                with open(dest_file_path, 'w') as out_file:
                    out_file.write(content)
                print(f"Decompressed and copied content from {src_file} to {dest_file_path}")
        else:
            print(f"File {src_file} does not exist")


def merge_text_files(data_path, output_file):
    """Merge multiple text files into a single JSON file."""
    text_files = glob.glob(os.path.join(data_path, '*.txt'))
    counter = 0

    with open(output_file, 'w', encoding='UTF-8') as outfile:
        for fname in text_files:
            counter += 1
            if counter % 1024 == 0:
                print(f"Merging file {counter}", flush=True)

            with open(fname, 'r', encoding='UTF-8') as infile:
                for row in infile:
                    json.dump({'text': row.strip()}, outfile)
                    outfile.write('\n')

    print(f"All files merged into {output_file}")


def tokenize_dataset(dataset, enc, num_proc=1):
    """Tokenize the dataset using GPT-2 encoding."""
    def process(example):
        ids = enc.encode_ordinary(example['text'])
        ids.append(enc.eot_token)  # Add the end of text token
        return {'ids': ids, 'len': len(ids)}

    tokenized = dataset.map(
        process,
        remove_columns=['text'],
        desc="Tokenizing dataset",
        num_proc=num_proc,
    )
    return tokenized


def save_tokenized_dataset(tokenized, output_dir):
    """Save tokenized dataset into binary files for training."""
    os.makedirs(output_dir, exist_ok=True)
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = os.path.join(output_dir, f'{split}.bin')
        dtype = np.uint16  # GPT-2's max token value < 2^16
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))

        idx = 0
        total_batches = 1024
        for batch_idx in tqdm(range(total_batches), desc=f'Writing {filename}'):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx:idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)

        arr.flush()


if __name__ == '__main__':
    # Set directories and parameters dynamically
    source_dir = '/data/openwebtext/unzip_folder'
    dest_dir = '/data/openwebtext/txt_folder'
    merged_output = '/data/openwebtext/merged_openwebtext.json'
    output_dir = '/data/openwebtext/tokenized_data'

    # Parameters for dataset processing
    num_proc = 8  # Number of workers for parallel processing
    encoding = tiktoken.get_encoding("gpt2")

    # Decompress .xz files
    decompress_xz_files(source_dir, dest_dir)

    # I prefer to merge txt into a single json file. You can load .txt directly of course.
    # Merge text files into a single JSON file
    merge_text_files(dest_dir, merged_output)

    # Load the dataset
    dataset = load_dataset("json", data_files=merged_output)
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test')  # Rename the test split to val

    # Tokenize the dataset
    tokenized_dataset = tokenize_dataset(split_dataset, encoding, num_proc)

    # Save tokenized dataset into binary files
    save_tokenized_dataset(tokenized_dataset, output_dir)
