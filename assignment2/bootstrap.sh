#!/bin/bash

readonly SCRIPT_NAME=$0
readonly DATASET_FILE=$1

readonly DATA_DIR="data"
readonly VIRTUALENV_DIR="venv"
readonly VIRTUALENV_PYTHON3_DIR="venv3"
readonly REQUIREMENTS_FILE="requirements.txt"
readonly REQUIREMENTS_PYTHON3_FILE="requirements_python3.txt"
readonly PYTHON_BIN="/usr/bin/python2.7"
readonly PYTHON3_BIN="/usr/bin/python3"

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
	rm -rf $VIRTUALENV_PYTHON3_DIR
}

create_virtualenv() {
	virtualenv --no-site-packages -p $PYTHON_BIN $VIRTUALENV_DIR
	virtualenv --no-site-packages -p $PYTHON3_BIN $VIRTUALENV_PYTHON3_DIR
}

save_dependencies_list() {
	local $file=$0
	rm -f $file
	pip freeze > $file
}

install_dependencies() {
	pip install numpy
	pip install scipy
	pip install ipython
	pip install sklearn
}

install_dependencies_from_requirements() {
	pip install -r $REQUIREMENTS_FILE
}

setup_dataset() {
	mkdir -p $DATA_DIR
	unzip $DATASET_FILE -d $DATA_DIR/
}

create_protcol2_dataset() {
	python cifar_for_python2.py
}

main() {
	reset_environment

	echo
	echo "[${SCRIPT_NAME}] Creating virtual environment..."
	create_virtualenv

	echo
	echo "[${SCRIPT_NAME}] Installing dependencies..."

	source $VIRTUALENV_DIR/bin/activate
	install_dependencies
	# alternatively, install from requirements.txt:
	# install_dependencies_from_requirements
	save_dependencies_list $REQUIREMENTS_FILE
	deactivate

	source $VIRTUALENV_PYTHON3_DIR/bin/activate
	install_dependencies
	# alternatively, install from requirements.txt:
	# install_dependencies_from_requirements
	save_dependencies_list $REQUIREMENTS_PYTHON3_FILE
	deactivate

	echo
	echo "[${SCRIPT_NAME}] Extracting dataset..."
	setup_dataset

	echo
	echo "[${SCRIPT_NAME}] Converting dataset for Python 2..."
	source $VIRTUALENV_PYTHON3_DIR/bin/activate
	create_protcol2_dataset
	deactivate

	echo
	echo "[${SCRIPT_NAME}] Done."
}
main
