import os
import glob

def clean_files(directory):
    extensions = [".hyps", ".refs", ".loss", ".bleu"]
    if not os.path.exists(directory):
        print(f"Directory does not exist: {directory}")
        return

    removed = 0
    for ext in extensions:
        pattern = os.path.join(directory, f"*{ext}")
        for file_path in glob.glob(pattern):
            os.remove(file_path)
            print(f"Deleted: {file_path}")
            removed += 1

    if removed == 0:
        print("No files matched for deletion.")
    else:
        print(f"{removed} files deleted.")
