#!/bin/bash

readonly SCRIPT_NAME=$0
readonly DATASET_FILE=$1

readonly CIFAR_DIR="cifar-2class-py2"
readonly LOGS_DIR="logs"
readonly VIRTUALENV_DIR="venv"
readonly REQUIREMENTS_FILE="requirements.txt"

usage() {
	echo "usage: $SCRIPT_NAME /path/to/cifar-2class-py2.zip"
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
	rm -rf $CIFAR_DIR
	rm -rf $VIRTUALENV_DIR
}

create_virtualenv() {
	virtualenv --no-site-packages $VIRTUALENV_DIR
}

save_dependencies_list() {
	local $file=$0
	rm -f $file
	pip freeze > $file
}

install_dependencies() {
	pip install argparse
	pip install numpy
	pip install scipy
	pip install ipython
	pip install sklearn
	pip install matplotlib
}

install_dependencies_from_requirements() {
	pip install -r $REQUIREMENTS_FILE
}

setup_dataset() {
	unzip $DATASET_FILE
	rm -rf __MACOSX
}

create_protcol2_dataset() {
	python cifar_for_python2.py
}

create_logs_dir() {
	mkdir -p $LOGS_DIR
}

main() {
	reset_environment

	echo
	echo "[${SCRIPT_NAME}] Creating virtual environment..."
	create_virtualenv

	echo
	echo "[${SCRIPT_NAME}] Installing python2.7 dependencies..."

	source $VIRTUALENV_DIR/bin/activate
	install_dependencies
	# alternatively, install from requirements.txt:
	# install_dependencies_from_requirements
	save_dependencies_list $REQUIREMENTS_FILE
	deactivate

	echo
	echo "[${SCRIPT_NAME}] Extracting dataset..."
	setup_dataset
	create_logs_dir

	echo
	echo "[${SCRIPT_NAME}] Done."
}
main
