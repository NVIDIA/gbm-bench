# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
	rm -f catboost_training.json
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
