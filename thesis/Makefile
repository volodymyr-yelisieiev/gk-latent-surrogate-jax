LATEXMK ?= latexmk
MAIN := main-thesis
BUILD := build

.PHONY: all pdf clean distclean

all: pdf

pdf:
	$(LATEXMK) -xelatex -interaction=nonstopmode -halt-on-error -file-line-error -outdir=$(BUILD) $(MAIN).tex

clean:
	$(LATEXMK) -C -outdir=$(BUILD) $(MAIN).tex

distclean: clean
	rm -f $(BUILD)/$(MAIN).pdf
