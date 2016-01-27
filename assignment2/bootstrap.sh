#!/bin/bash

readonly SCRIPT_NAME=$0
readonly DATASET_FILE=$1

readonly DATA_DIR="data"
readonly VIRTUALENV_DIR="venv"
readonly REQUIREMENTS_FILE="requirements.txt"

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
	virtualenv -p /usr/bin/python2.7 --no-site-packages $VIRTUALENV_DIR
}

install_dependencies() {
	pip install numpy
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
	# install_dependencies_from_requirements
	install_dependencies

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
