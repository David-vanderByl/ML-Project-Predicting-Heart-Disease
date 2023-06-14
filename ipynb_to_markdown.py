import os
import re
import subprocess
import shutil

# Specify the path to the .ipynb file
ipynb_file = 'Predicting Heart Disease.ipynb'
base_name = os.path.splitext(ipynb_file)[0]

# Replace spaces with underscores in the base_name
base_name = base_name.replace(' ', '_')

# The name of the subfolder to put the plots in
plots_folder = f"{base_name}_plots"

# Make the subfolder
os.makedirs(plots_folder, exist_ok=True)

# Run nbconvert via command line to convert the notebook to markdown
subprocess.run(["jupyter", "nbconvert", "--to", "markdown", "--output", "README", ipynb_file])

# The markdown file produced by nbconvert
markdown_file = "README.md"

# Rename output images and update the image references in the markdown file
with open(markdown_file, 'r', encoding='utf-8') as f:
    markdown = f.read()

# Find all image references
image_refs = re.findall(r'\!\[png\]\((.+)\)', markdown)

for i, image_ref in enumerate(image_refs):
    old_image_path = image_ref
    new_image_name = f"output_{i}.png"
    new_image_path = os.path.join(plots_folder, new_image_name)

    # Rename (move) the image
    os.rename(old_image_path, new_image_path)

    # Update the image reference in the markdown
    markdown = markdown.replace(old_image_path, new_image_path)

# Save the updated markdown back to the README.md file
with open('README.md', 'w', encoding='utf-8') as f:
    f.write(markdown)

# Remove the README_files folder
shutil.rmtree('README_files', ignore_errors=True)
