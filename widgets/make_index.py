import os
import argparse
import html # For escaping special characters in titles/paths if needed

def generate_pretty_name(filename):
    """Generates a human-readable name from a filename."""
    name_without_ext = os.path.splitext(filename)[0]
    # Replace hyphens and underscores with spaces, then title case
    return name_without_ext.replace('-', ' ').replace('_', ' ').title()

def create_html_index(scan_dir, output_filename="index.html"):
    """
    Scans a directory for HTML files and creates an index.html file.

    Args:
        scan_dir (str): The directory to scan.
        output_filename (str): The name of the index file to create (e.g., "index.html").
    """
    abs_scan_dir = os.path.abspath(scan_dir)
    if not os.path.isdir(abs_scan_dir):
        print(f"Error: Directory '{scan_dir}' not found.")
        return

    html_files_map = {} # { 'relative_dir_path': ['file1.html', 'file2.html'], ... }

    # Walk through the directory
    for root, dirs, files in os.walk(abs_scan_dir, topdown=True):
        # Optional: Exclude hidden directories (like .git, .vscode)
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        # Filter for HTML files, excluding the output_filename itself
        current_dir_html_files = sorted([
            f for f in files 
            if f.lower().endswith(".html") and f.lower() != output_filename.lower()
        ])

        if current_dir_html_files:
            # Get the path relative to the scan_dir
            # If root is abs_scan_dir, rel_dir will be '.'
            rel_dir = os.path.relpath(root, abs_scan_dir)
            html_files_map[rel_dir] = current_dir_html_files

    if not html_files_map:
        print(f"No HTML files found in '{scan_dir}' (excluding '{output_filename}').")
        # Optionally, create an empty index or an index saying "No files found"
        # For now, we'll just return
        return

    # Start building HTML content
    html_content = []
    html_content.append("<!DOCTYPE html>")
    html_content.append("<html lang=\"en\">")
    html_content.append("<head>")
    html_content.append("    <meta charset=\"UTF-8\">")
    html_content.append("    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">")
    html_content.append(f"    <title>Index of {html.escape(os.path.basename(abs_scan_dir))}</title>")
    html_content.append("    <style>")
    html_content.append("        body { font-family: sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; }")
    html_content.append("        h1 { color: #333; border-bottom: 2px solid #ccc; padding-bottom: 10px; }")
    html_content.append("        h2 { color: #555; margin-top: 30px; border-bottom: 1px dashed #ddd; padding-bottom: 5px; }")
    html_content.append("        ul { list-style-type: none; padding-left: 0; }")
    html_content.append("        li { margin-bottom: 8px; background-color: #fff; padding: 10px; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }")
    html_content.append("        a { text-decoration: none; color: #007bff; }")
    html_content.append("        a:hover { text-decoration: underline; }")
    html_content.append("        .directory-path { font-size: 0.9em; color: #777; margin-left: 5px; }")
    html_content.append("    </style>")
    html_content.append("</head>")
    html_content.append("<body>")
    html_content.append(f"    <h1>Index of HTML Files in '{html.escape(os.path.basename(abs_scan_dir))}'</h1>")

    # Sort directories: root ('.') first, then alphabetically
    sorted_dirs = sorted(html_files_map.keys(), key=lambda x: (x == '.', x))

    for rel_dir_path in sorted_dirs:
        files = html_files_map[rel_dir_path]
        
        if rel_dir_path == ".":
            # Files in the root of scan_dir
            if files: # Ensure there are actually files before printing the heading
                html_content.append("    <h2>Root Files</h2>")
        else:
            html_content.append(f"    <h2>Directory: {html.escape(rel_dir_path)}</h2>")
        
        html_content.append("    <ul>")
        for filename in files:
            # Path for the link, relative to the index.html file
            if rel_dir_path == ".":
                link_path = filename
            else:
                link_path = os.path.join(rel_dir_path, filename).replace(os.sep, '/') # Ensure forward slashes for web
            
            pretty_name = generate_pretty_name(filename)
            html_content.append(f"        <li><a href=\"{html.escape(link_path)}\">{html.escape(pretty_name)}</a> <span class='directory-path'>({html.escape(filename)})</span></li>")
        html_content.append("    </ul>")

    html_content.append("</body>")
    html_content.append("</html>")

    # Write the index file
    output_file_path = os.path.join(abs_scan_dir, output_filename)
    try:
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(html_content))
        print(f"Successfully created '{output_file_path}'")
    except IOError as e:
        print(f"Error writing to '{output_file_path}': {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an index.html for HTML files in a directory.")
    parser.add_argument("directory", nargs="?", default=".",
                        help="The directory to scan (default: current directory).")
    parser.add_argument("-o", "--output", default="index.html",
                        help="The name of the output HTML index file (default: index.html).")
    
    args = parser.parse_args()
    
    create_html_index(args.directory, args.output)
