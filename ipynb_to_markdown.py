from nbconvert import MarkdownExporter
import nbformat

# Specify the path to the .ipynb file
ipynb_file = 'Predicting Heart Disease.ipynb'

# Load the notebook file
with open(ipynb_file, 'r', encoding='utf-8') as f:
    notebook = nbformat.read(f, as_version=4)

# Configure the exporter
exporter = MarkdownExporter()
exporter.exclude_output_prompt = False  # Include code cell prompts
exporter.exclude_input = True  # Exclude code cell inputs
exporter.exclude_output = False  # Include code cell outputs

# Convert the notebook to Markdown
markdown, _ = exporter.from_notebook_node(notebook)

# Save the Markdown content to a file
markdown_file = 'markdown.md'
with open(markdown_file, 'w', encoding='utf-8') as f:
    f.write(markdown)

