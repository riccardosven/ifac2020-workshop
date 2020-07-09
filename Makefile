
env-create:
	python -m venv .venv --prompt workshop
	make env-update

env-update:
	bash -c ". .venv/bin/activate; \
		pip install --upgrade -r requirements.txt"

check: reformat lint

clean:
	rm -r -f .venv

reformat:
	isort --recursive ./
	black ./

lint:
	python -m pycodestyle . --exclude '.venv'
	isort --recursive --check-only ./
	black --check ./
	pylint ./
	mypy ./
    
