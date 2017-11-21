
DATASET  ?= football

default:
	@echo "make what? Available targets are:"
	@echo "  . runAll - run benchmark on all current datasets"
	@echo "  . run    - run benchmark on a given dataset"
	@echo "Variables are:"
	@echo "  . DATASET - name of the dataset"

runAll: warmUp
	$(MAKE) -f ./gbm-perf/Makefile DATASET=airline    run
	$(MAKE) -f ./gbm-perf/Makefile DATASET=bosch      run
	$(MAKE) -f ./gbm-perf/Makefile DATASET=football   run
	$(MAKE) -f ./gbm-perf/Makefile DATASET=fraud      run
	$(MAKE) -f ./gbm-perf/Makefile DATASET=higgs      run
	$(MAKE) -f ./gbm-perf/Makefile DATASET=msltr      run
	$(MAKE) -f ./gbm-perf/Makefile DATASET=msltr_full run
	$(MAKE) -f ./gbm-perf/Makefile DATASET=planet     run

# make sure that opencl kernels for LightGBM are all compiled!
warmUp:
	$(MAKE) -f ./gbm-perf/Makefile DATASET=football run

run:
	./gbm-perf/runme.py -dataset $(DATASET) -root ./gbm-datasets
