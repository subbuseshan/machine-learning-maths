ENV := ml-maths
RUN := conda run --no-capture-output -n $(ENV)

.PHONY: help setup update lab smoke clean

help:
	@echo "make setup   - create the '$(ENV)' conda env from environment.yml"
	@echo "make update  - sync the env after editing environment.yml"
	@echo "make lab     - launch JupyterLab at the repo root"
	@echo "make smoke   - check every notebook dependency imports"
	@echo "make clean   - delete __pycache__ and .ipynb_checkpoints"

setup:
	conda env create -f environment.yml
	$(RUN) python -m ipykernel install --user --name $(ENV) --display-name "Python ($(ENV))"

update:
	conda env update -f environment.yml --prune

lab:
	$(RUN) jupyter lab

smoke:
	$(RUN) python -c "import numpy, scipy, pandas, matplotlib, seaborn, sklearn, sympy, cv2, jax, ipywidgets, ipympl; print('all imports OK')"

clean:
	find . \( -name __pycache__ -o -name .ipynb_checkpoints \) -type d -prune -exec rm -rf {} +
