.PHONY: install test notebooks docs serve clean

install:
	pip install -e ".[dev,docs]"

test:
	pytest

# great-docs renders .qmd, not .ipynb: the tutorial is converted first. Its own
# front matter carries `execute: enabled: false` (it shells out to NAIF and never
# stored outputs), so the page shows the commands unrun.
notebooks:
	@for nb in notebooks/*.ipynb; do \
		[ -e "$$nb" ] || continue; \
		quarto convert "$$nb" --output "docs/07-$$(basename "$${nb%.ipynb}").qmd"; \
	done

docs: notebooks
	great-docs build

serve: notebooks
	great-docs preview

clean:
	rm -rf dist build *.egg-info great-docs/ docs/07-*.qmd
