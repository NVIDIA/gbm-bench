
DATASET  ?= football
MAXDEPTH ?= 5
NTREES   ?= 100

default:
	@echo "make what? Available targets are:"
	@echo " . runAll - run benchmark on all current datasets"
	@echo " . run    - run benchmark on a given dataset"
	@echo "    Flags to customize:"
	@echo "    . DATASET  - name of the dataset"
	@echo "    . MAXDEPTH - max depth of trees"
	@echo "    . NTREES   - number of trees in the forest"

runAll:
	$(MAKE) _runAll 2>&1 | tee output_$(MAXDEPTH)_$(NTREES).log

_runAll: warmUp
	$(MAKE) DATASET=airline    run
	$(MAKE) DATASET=bosch      run
	$(MAKE) DATASET=football   run
	$(MAKE) DATASET=fraud      run
	$(MAKE) DATASET=higgs      run
	$(MAKE) DATASET=msltr      run
	$(MAKE) DATASET=msltr_full run
	$(MAKE) DATASET=planet     run
	./json2csv.py *.json > benchmark_$(MAXDEPTH)_$(NTREES).csv
	rm -f *.json

# make sure that opencl kernels for LightGBM are all compiled!
warmUp:
	$(MAKE) DATASET=football   run

run:
	rm -f $(DATASET).json
	./runme.py -dataset $(DATASET) \
	           -root ../gbm-datasets \
	           -output $(DATASET).json \
	           -maxdepth $(MAXDEPTH) \
	           -ntrees $(NTREES)
