import hashlib
import sys


def sha256_hash_file(filename):
    sha256_hash = hashlib.sha256()
    with open(filename, "rb") as file:
        # Read and update hash string in chunks to handle large files
        for byte_block in iter(lambda: file.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


# Usage example
filename = sys.argv[1]
print(sha256_hash_file(filename))
