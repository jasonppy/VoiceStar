emilia_root=$1 # /data/scratch/pyp/datasets/emilia/downloads
for file in ${emilia_root}/* ; do
# Check if gzip compressed archive
if file "$file" | grep -q 'gzip compressed data'; then
    # Extract string of form 'was "EN_B00100.tar"'' from the output of the file command to keep EN_B00100
    filename=$(file "$file" | grep -oP '(?<=was ")[^"]*' | sed 's/\.tar$//')
    # Get the file size
    size=$(du -sh "$file" | cut -f1)
    original_filename=$(basename "$file")
    # Get URL string from corresponding JSON file with same basename
    json_file=$file.json
    if [ -f "$json_file" ]; then
        # url=$(jq -r '.url' "$json_file") # jq is not installed on the server
        url=$(python3 -c "import sys, json; print(json.load(open('$json_file'))['url'])")
    else
        url="N/A"
    fi
    # Compute SHA256 hash of the file
    hash=$(python sha256hash.py "$file")
    echo $original_filename
    # Write filename, size, hash, original filename, URL to output file
    echo "$filename, $size, $hash, $original_filename, $url" >> file_log.txt
fi
done

# Sort the output file by filename
sort -o file_log.txt file_log.txt