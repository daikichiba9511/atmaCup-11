
PACKAGE = loguru wandb flake8 mypy black pyyaml pytorch-lightning

config:
	@sh config.sh

develop:
	pip3 install -q -U ${PACKAGE}