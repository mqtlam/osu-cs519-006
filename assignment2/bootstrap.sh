#!/bin/bash

readonly SCRIPT_NAME=$0
readonly DATASET_FILE=$1

readonly DATA_DIR="data"
readonly VIRTUALENV_DIR="venv"
# readonly VIRTUALENV_DIR="venv3"
readonly REQUIREMENTS_FILE="requirements.txt"
readonly PYTHON_BIN="/usr/bin/python2.7"
# readonly PYTHON_BIN="/usr/bin/python3"

usage() {
	echo "usage: $SCRIPT_NAME /path/to/cifar-2class-py.zip"
	echo ""
	echo "Set up everything for assignment:"
	echo -e "\t- Set up virtual environment"
	echo -e "\t- Install Python dependencies"
	echo -e "\t- Extract dataset"
}
if [ "$#" -ne 1 ]; then
	echo "Invalid number of arguments."
	usage
	exit 1
fi

reset_environment() {
	rm -rf $DATA_DIR
	rm -rf $VIRTUALENV_DIR
}

create_virtualenv() {
	virtualenv --no-site-packages -p $PYTHON_BIN $VIRTUALENV_DIR
}

save_dependencies_list() {
	rm -f $REQUIREMENTS_FILE
	pip freeze > $REQUIREMENTS_FILE
}

install_dependencies() {
	pip install numpy
	pip install scipy
	pip install ipython

	save_dependencies_list
}

install_dependencies_from_requirements() {
	pip install -r $REQUIREMENTS_FILE
}

setup_dataset() {
	mkdir -p $DATA_DIR
	unzip $DATASET_FILE -d $DATA_DIR/
}

main() {
	reset_environment

	echo
	echo "[${SCRIPT_NAME}] Creating virtual environment..."
	create_virtualenv
	source $VIRTUALENV_DIR/bin/activate

	echo
	echo "[${SCRIPT_NAME}] Installing dependencies..."
	install_dependencies
	# alternatively, install from requirements.txt:
	# install_dependencies_from_requirements

	echo
	echo "[${SCRIPT_NAME}] Extracting dataset..."
	setup_dataset

	echo
	echo "[${SCRIPT_NAME}] Cleaning up..."
	deactivate

	echo
	echo "[${SCRIPT_NAME}] Done."
}
main
