#!/usr/bin/env python3.11
import sys
import re

def fix_patch_file(file_path):
    """
    Reads a patch file, recounts lines in each hunk, and prints the corrected patch.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}", file=sys.stderr)
        return

    hunk_header_re = re.compile(r'^@@ -(\d+)(,\d+)? \+(\d+)(,\d+)? @@.*')
    hunk_lines = []
    current_hunk_header = None
    
    def process_hunk():
        if not current_hunk_header:
            return

        # Recount lines
        original_len = sum(1 for line in hunk_lines if line.startswith((' ', '-')))
        modified_len = sum(1 for line in hunk_lines if line.startswith((' ', '+')))

        # Get start lines from the original header
        match = hunk_header_re.match(current_hunk_header)
        start_orig = match.group(1)
        start_mod = match.group(3)
        
        # Format new header
        # Handle cases where count is 1 (e.g., @@ -1 +1,2 @@)
        orig_count_str = f",{original_len}" if original_len != 1 else ""
        mod_count_str = f",{modified_len}" if modified_len != 1 else ""

        new_header = f"@@ -{start_orig}{orig_count_str} +{start_mod}{mod_count_str} @@\n"
        
        # Print the corrected hunk
        sys.stdout.write(new_header)
        sys.stdout.writelines(hunk_lines)

    for line in lines:
        if line.startswith('@@'):
            # Process the previous hunk before starting a new one
            process_hunk()
            
            # Start a new hunk
            current_hunk_header = line
            hunk_lines = []
        elif current_hunk_header:
            hunk_lines.append(line)
        else:
            # Print lines before the first hunk (e.g., ---, +++)
            sys.stdout.write(line)

    # Process the last hunk in the file
    process_hunk()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <path_to_patch_file>")
        sys.exit(1)
    
    fix_patch_file(sys.argv[1])
