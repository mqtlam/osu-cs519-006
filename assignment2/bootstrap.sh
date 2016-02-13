#!/bin/bash
# This script should be run first before everything else
# to set up the virtual environment, dataset, folders, etc.

# global constants:

# where to look for the dataset after extracting from zip
readonly CIFAR_DIR="cifar-2class-py2"
# for logs generated from main.py
readonly LOGS_DIR="logs"
# for logs generated from cluster jobs (hyperparmameters.py)
readonly JOBS_DIR="jobs"
# for figures generated from analyze_results.py
readonly FIGURES_DIR="figures"
# virtual environment directory
readonly VIRTUALENV_DIR="venv"
# requirements file for generating a list of Python dependencies
readonly REQUIREMENTS_FILE="requirements.txt"

# check arguments
readonly SCRIPT_NAME=$0
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

# command line arguments
readonly DATASET_FILE=$1

reset_environment() {
	# reset by deleting folders
	rm -rf $CIFAR_DIR
	rm -rf $VIRTUALENV_DIR
}

create_virtualenv() {
	# create virtual environment
	virtualenv --no-site-packages $VIRTUALENV_DIR
}

save_dependencies_list() {
	# save the Python dependencies to a requirements file
	# (not really useful except for knowing all the dependencies)
	local $file=$0
	rm -f $file
	pip freeze > $file
}

install_dependencies() {
	# install the Python dependencies using pip (inside virtual environment)
	pip install argparse
	pip install numpy
	pip install scipy
	pip install ipython
	pip install sklearn
	pip install matplotlib
}

install_dependencies_from_requirements() {
	# could just install from the requirements file
	# instead of calling install_dependencies
	pip install -r $REQUIREMENTS_FILE
}

setup_dataset() {
	# set up the dataset by extracting the dataset zip file
	unzip $DATASET_FILE
	rm -rf __MACOSX
}

create_dirs() {
	# create empty directories used later
	mkdir -p $LOGS_DIR
	mkdir -p $JOBS_DIR
	mkdir -p $FIGURES_DIR
}

main() {
	# main script
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
	create_dirs

	echo
	echo "[${SCRIPT_NAME}] Done."
}
main
