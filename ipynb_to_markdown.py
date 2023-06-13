from nbconvert import MarkdownExporter
import nbformat
import shutil
import os

# Specify the path to the .ipynb file
ipynb_file = 'Predicting Heart Disease.ipynb'

# Load the notebook file
with open(ipynb_file, 'r', encoding='utf-8') as f:
    notebook = nbformat.read(f, as_version=4)

# Configure the exporter
exporter = MarkdownExporter()

# Convert the notebook to Markdown
markdown, resources = exporter.from_notebook_node(notebook)

# Save the Markdown content to a file
markdown_file = 'markdown.md'
with open(markdown_file, 'w', encoding='utf-8') as f:
    f.write(markdown)

# Create a directory for saving the output images
output_dir = 'output_images'
os.makedirs(output_dir, exist_ok=True)

# Find and save output images
for output_file, output_data in resources.get('outputs', {}).items():
    if isinstance(output_data, dict) and 'image/png' in output_data.get('output_type', []):
        image_data = output_data.get('data', {}).get('image/png', b'')
        if image_data:
            image_filename = f'{output_file}.png'
            image_path = os.path.join(output_dir, image_filename)
            with open(image_path, 'wb') as f:
                f.write(image_data)

# Update the image paths in the Markdown file
for image_file in os.listdir(output_dir):
    image_filename = os.path.basename(image_file)
    new_image_path = f'{output_dir}/{image_filename}'
    markdown = markdown.replace(image_filename, new_image_path)

# Save the updated Markdown file
with open(markdown_file, 'w', encoding='utf-8') as f:
    f.write(markdown)
