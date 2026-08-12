LATEXMK ?= latexmk
MAIN := main-thesis
BUILD := build

.PHONY: all pdf audit clean distclean

all: pdf

pdf:
	$(LATEXMK) -xelatex -interaction=nonstopmode -halt-on-error -file-line-error -outdir=$(BUILD) $(MAIN).tex

audit: pdf
	python3 scripts/audit_pdf.py --pdf $(BUILD)/$(MAIN).pdf --log $(BUILD)/$(MAIN).log --output $(BUILD)/manuscript-audit.json

clean:
	$(LATEXMK) -C -outdir=$(BUILD) $(MAIN).tex

distclean: clean
	rm -f $(BUILD)/$(MAIN).pdf
