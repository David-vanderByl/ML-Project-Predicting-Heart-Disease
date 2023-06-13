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

# Check if 'jupyter' key exists in metadata
if 'jupyter' not in notebook['metadata']:
    # If 'jupyter' key is missing, add it with an empty dictionary
    notebook['metadata']['jupyter'] = {}

# Add nbconvert and widgets configuration to 'jupyter' metadata
notebook['metadata']['jupyter']['nbconvert'] = {'execute_notebooks': 'auto'}
notebook['metadata']['jupyter']['widgets'] = {
    'widget_state': {},
    'application/vnd.jupyter.widget-view+json': {},
    'version_major': 2,
    'version_minor': 0
}

# Convert the notebook to Markdown
markdown, _ = exporter.from_notebook_node(notebook, resources={})

# Save the Markdown content to a file
markdown_file = 'markdown.md'
with open(markdown_file, 'w', encoding='utf-8') as f:
    f.write(markdown)
