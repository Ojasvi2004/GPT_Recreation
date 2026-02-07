import numpy as np
import tiktoken
import os

def create_bin_from_txt(txt_file, bin_file, chunk_size=10_000):
    """
    Convert a large text file into a memory-mapped .bin of tokens without loading full file into RAM.

    Args:
        txt_file (str): Path to the text file.
        bin_file (str): Path to save the .bin file.
        chunk_size (int): Number of characters to read per chunk.
    """
    enc = tiktoken.get_encoding('gpt2')

    # Delete existing bin file if exists
    if os.path.exists(bin_file):
        os.remove(bin_file)

    with open(txt_file, "r", encoding="utf-8") as f, open(bin_file, "ab") as out:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            tokens = enc.encode(chunk)
            np.array(tokens, dtype=np.uint16).tofile(out)  # write to open file object
    print(f"Finished writing tokens to {bin_file}")


# Example usage
create_bin_from_txt("TinyStoriesV3-GPT4-train.txt", "train.bin")
create_bin_from_txt("TinyStoriesV3-GPT4-train.txt", "val.bin")
