import sys
from .clean import clean_files
from .build_llama_vocab import build_llama_vocab

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 -m scripts <command> <target_directory>")
        sys.exit(1)

    command = sys.argv[1]
    target_dir = sys.argv[2]

    if command == "clean":
        clean_files(target_dir)
    elif command == "build_llama_vocab":
        build_llama_vocab(target_dir)
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()
