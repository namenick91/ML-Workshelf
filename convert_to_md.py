import nbformat
from nbconvert import MarkdownExporter

# name = 'dl_cv_classification_simpsons'
# name = 'dl_semantic_segmentation_medical'
# name = 'ml_classification_churn'
name = 'ml_classification_game_of_thrones'

nb = nbformat.read(f"./{name}/main.ipynb", as_version=4)
body, resources = MarkdownExporter().from_notebook_node(nb)

with open(f"{name}.md", "w", encoding="utf-8") as f:
    f.write(body)

# resources["outputs"] contains extracted images if you want to handle them yourself
