#!/bin/bash
# This script should be run first before everything else
# to set up the virtual environment, dataset, folders, etc.

# global constants:

# virtual environment directory
readonly VIRTUALENV_DIR="venv"
# requirements file for generating a list of Python dependencies
readonly REQUIREMENTS_FILE="requirements.txt"

# check arguments
readonly SCRIPT_NAME=$0
usage() {
	echo "usage: $SCRIPT_NAME"
	echo ""
	echo "Set up everything for assignment:"
	echo -e "\t- Set up virtual environment"
	echo -e "\t- Install Python dependencies"
}

reset_environment() {
	# reset by deleting folders
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
	pip install numpy
	pip install git+git://github.com/Theano/Theano.git
	pip install keras
	pip install ipython
}

install_dependencies_from_requirements() {
	# could just install from the requirements file
	# instead of calling install_dependencies
	pip install -r $REQUIREMENTS_FILE
}

main() {
	# main script
	reset_environment

	echo
	echo "[${SCRIPT_NAME}] Creating virtual environment..."
	create_virtualenv

	echo
	echo "[${SCRIPT_NAME}] Installing python dependencies..."

	source $VIRTUALENV_DIR/bin/activate
	install_dependencies
	# alternatively, install from requirements.txt:
	# install_dependencies_from_requirements
	save_dependencies_list $REQUIREMENTS_FILE
	deactivate

	echo
	echo "[${SCRIPT_NAME}] Done."
}
main
