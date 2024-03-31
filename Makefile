all: run_gen_readme

run_gen_readme:
	python3 tools/replace_image.py
	python3 tools/gen_readme.py

