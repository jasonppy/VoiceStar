#!/bin/bash

# Define the root directory where the tar files are located
root=${root:-/data/scratch/pyp/datasets/emilia/downloads} # Example: /data/scratch/pyp/datasets/emilia/downloads
exist_log_file="file_log_debug.txt" # Log of files to delete
delete_log="deleted_files.log" # Log of successfully deleted files
error_log="delete_errors.log"  # Log of errors (e.g., missing files)

# Clear previous logs
> "$delete_log"
> "$error_log"

echo "Starting deletion of tar files listed in $exist_log_file..."

# Loop through each line in exist_log_file
while IFS=',' read -r filename size local_sha256 original_filename url; do
    # Trim leading/trailing whitespace
    original_filename=$(echo "$original_filename" | xargs)

    # Construct the full path to the tar file
    tar_file="${root}/${original_filename}"

    # Check if the tar file exists
    if [ -f "$tar_file" ]; then
        # Attempt to delete the file
        if rm -f "$tar_file"; then
            echo "✅ Deleted: $tar_file"
            echo "$tar_file" >> "$delete_log"
        else
            echo "❌ Failed to delete: $tar_file"
            echo "$tar_file" >> "$error_log"
        fi
    else
        # Log missing files
        echo "❌ File not found: $tar_file"
        echo "$tar_file" >> "$error_log"
    fi
done < "$exist_log_file"

echo "Deletion process completed."
echo "Deleted files are logged in $delete_log."
echo "Errors (if any) are logged in $error_log."