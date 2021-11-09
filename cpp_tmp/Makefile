QT_DIR := /usr/lib/qt5/bin/


.PHONY: app
app: | $(QT_DIR)
	(mkdir -p build && cd build && $(QT_DIR)qmake ../src/sfc_fuzzy_kmeans.pro && make)

.PHONY: all
all: app

.PHONY: run
run: app
	cd bin && ./sfc_fuzzy_kmeans


.PHONY: clean
clean:
	rm -rf build
	rm bin/sfc_fuzzy_kmeans
