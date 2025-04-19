# include only the following directories in a zip file:
# - models/model_B
# - final_dataset_scripts
# - dataset/final_datasets
# - requirements.txt
# - .gitignore
# - utils

import os
import zipfile


def format_size(size):
    """Convert bytes to human-readable format"""
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} TB"


def create_zip():
    zip_filename = "sepsis_upload.zip"

    # Items to include in the zip
    items_to_include = [
        "models/model_B",
        "final_dataset_scripts",
        "dataset/final_datasets",
        "requirements.txt",
        ".gitignore",
        "utils",
        # Removed .git as it likely contains large history files
    ]

    # Directories to exclude
    exclude_dirs = ["models/model_B/results", "models/model_B/saved"]
    exclude_patterns = [
        ".git",
        "__pycache__",
        "*.pyc",
        "*.pyo",
        "*.pyd",
        "*.so",
        "*.dylib",
        ".DS_Store",
    ]

    total_size = 0
    file_count = 0

    # Create zip file
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for item in items_to_include:
            try:
                if os.path.exists(item):
                    if os.path.isfile(item):
                        # Add file directly with clean path
                        file_size = os.path.getsize(item)
                        zipf.write(item, item)
                        total_size += file_size
                        file_count += 1
                        print(f"Added file: {item} ({format_size(file_size)})")
                    elif os.path.isdir(item):
                        # Add directory and its contents
                        for root, dirs, files in os.walk(item):
                            # Skip excluded directories
                            pass

                            # Filter directories to avoid traversing excluded ones
                            dirs[:] = [
                                d
                                for d in dirs
                                if not any(
                                    excl in os.path.join(root, d)
                                    for excl in exclude_dirs
                                )
                                and not any(
                                    d.endswith(pat.strip("*"))
                                    for pat in exclude_patterns
                                    if pat.endswith("*")
                                )
                            ]

                            # Skip this directory if it matches exclusion patterns
                            if any(excl in root for excl in exclude_dirs):
                                continue

                            for file in files:
                                # Skip files matching exclusion patterns
                                if any(
                                    file.endswith(pat.strip("*"))
                                    for pat in exclude_patterns
                                    if pat.endswith("*")
                                ):
                                    continue

                                file_path = os.path.join(root, file)
                                file_size = os.path.getsize(file_path)

                                # Store file with its relative path for proper extraction
                                zipf.write(file_path, file_path)
                                total_size += file_size
                                file_count += 1

                                if (
                                    file_size > 50 * 1024 * 1024
                                ):  # Files larger than 50MB
                                    print(
                                        f"Large file: {file_path} ({format_size(file_size)})"
                                    )

                        print(f"Added directory and contents: {item}")
                else:
                    print(f"Warning: {item} does not exist and will be skipped")
            except Exception as e:
                print(f"Error adding {item}: {e}")

    final_size = os.path.getsize(zip_filename)
    print(f"Zip file created: {zip_filename}")
    print(f"Files added: {file_count}")
    print(f"Total uncompressed size: {format_size(total_size)}")
    print(f"Final zip size: {format_size(final_size)}")
    print(f"Compression ratio: {total_size/final_size:.2f}x")

    return zip_filename


if __name__ == "__main__":
    create_zip()
