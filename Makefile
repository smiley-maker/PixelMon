# Define the number of epochs to train
NUM_EPOCHS=150000

# Define ANSI escape sequences for colors
GREEN := $(shell tput -Txterm setaf 2)
RESET := $(shell tput -Txterm sgr0)
PURPLE := $(shell tput -Txterm setaf 5)


#  This target removes build artifacts and directories
#  generated during the build process.
.PHONY: clean_target
clean_target:
	$(RM) build dist *.egg target
	@echo "$(GREEN)Build Directories Cleaned Successfully$(RESET)"



#  This target installs project dependencies listed in 
#  'requirements.txt' into the virtual environment.
.PHONY: install
install: activate_venv requirements.txt requirements-mac-metal.txt
ifeq ($(shell uname),Darwin)
	$(BIN)$(PIP) install -r requirements-mac-metal.txt
else
	$(BIN)$(PIP) install -r requirements.txt
endif
	@echo "$(GREEN)Dependencies Installed Successfully$(RESET)"


#  This target installs the project in 'editable' mode, 
#  allowing changes to the source to be immediately reflected 
#  without needing to reinstall the package.
.PHONY: compile
compile: setup.py
	$(BIN) $(PYTHON) -m pip install -e .


#  This target installs dependencies, compiles the project, 
#  and runs all unit tests in src/tests directory
.PHONY: test
test: install compile
	$(TOX)
	@echo "$(GREEN)Completed Tests$(RESET)"


# Runs necessary scripts to start Pokemon generation training, 
# model saving, and testing. *Work in progress
.PHONY: train_pokemon
train_pokemon: install compile
	$(BIN)$(PYTHON) src/pipelines/pokemon_pipeline.py

#  This target cleans the target directories, installs dependencies, 
#  compiles the project, and builds distribution packages using Python's 
#  build module.
.PHONY: build
build: clean_target install compile setup.py
	$(BIN)$(PYTHON) -m build --outdir target/dist
	@echo "$(GREEN)Project Built Complete$(RESET)"