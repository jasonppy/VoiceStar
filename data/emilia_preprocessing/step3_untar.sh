#!/bin/bash

# Define the root directory where the tar files are located
root=$1 # /data/scratch/pyp/datasets/emilia/downloads
save_root=$2 # /data/scratch/pyp/datasets/emilia/preprocessed/audio

mkdir -p "${save_root}"

# Input log files
log_file="file_log.txt"       # Full log of files to process
exist_log_file="file_log_debug.txt" # Log of already processed files
failure_log="untar_failures.log" # Log file for untar failures

# Clear previous failure log
> "$failure_log"

# Create an array of filenames already processed (from exist_log_file)
if [ -f "$exist_log_file" ]; then
    mapfile -t existing_files < "$exist_log_file"
else
    existing_files=()
fi

# Create a temporary filtered log of files to process
filtered_log="filtered_file_log.txt"
grep -v -F -f "$exist_log_file" "$log_file" > "$filtered_log"

# Count total filtered files
total_files=$(wc -l < "$filtered_log")
echo "Found $total_files entries to process in $filtered_log."

# Print the filtered files
echo "Filtered files to process:"
cat "$filtered_log"
echo

# Confirm before starting processing
read -p "Do you want to proceed with the above files? (y/n): " confirm
if [[ "$confirm" != "y" ]]; then
    echo "Operation canceled."
    rm -f "$filtered_log"
    exit 1
fi

# Start time
start_time=$(date +%s)

# Counter for how many lines we've processed
count=0

# Process filtered log
while IFS=',' read -r filename size local_sha256 original_filename url; do
    count=$((count + 1))

    # Trim leading/trailing whitespace
    filename=$(echo "$filename" | xargs)
    size=$(echo "$size" | xargs)
    local_sha256=$(echo "$local_sha256" | xargs)
    original_filename=$(echo "$original_filename" | xargs)
    url=$(echo "$url" | xargs)

    # Construct the full path to the tar file
    tar_file="${root}/${original_filename}"

    # Check if the tar file exists
    if [ ! -f "$tar_file" ]; then
        echo "❌ File not found: $tar_file"
        echo "$filename, $size, $local_sha256, $original_filename, $url" >> "$failure_log"
    else
        # Try to untar the file
        echo "[$count/$total_files] Untarring $tar_file..."
        
        if ! tar -xf "$tar_file" -C "${save_root}"; then
            # If untar fails, log the failure
            echo "❌ Failed to untar: $tar_file"
            echo "$filename, $size, $local_sha256, $original_filename, $url" >> "$failure_log"
        else
            echo "✅ Successfully untarred: $tar_file"
            # Append successfully untarred filename to exist_log_file
            echo "$filename" >> "$exist_log_file"
        fi
    fi

    # Calculate elapsed time, average time per file, and ETA
    now=$(date +%s)
    elapsed=$(( now - start_time ))  # total seconds since the start
    if [ $count -gt 0 ]; then
        avg_time=$(awk "BEGIN { printf \"%.2f\", $elapsed / $count }")
        remain=$(( total_files - count ))
        eta_seconds=$(awk "BEGIN { printf \"%.0f\", $avg_time * $remain }")
        eta_formatted=$(date -ud "@${eta_seconds}" +'%H:%M:%S')
        echo "Elapsed: ${elapsed}s | Avg/f: ${avg_time}s | Remaining: $remain files | ETA: ~${eta_formatted}"
    fi

done < "$filtered_log"

# Clean up temporary filtered log
rm -f "$filtered_log"

# Summary
echo "Untar operation completed. Check $failure_log for any failures."