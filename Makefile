AUTOPILOT_ROOT :=/tools/software/xilinx/Vitis_HLS/2021.1

CPPFLAGS += -I "${AUTOPILOT_ROOT}/include"
CPPFLAGS += -D__SIM_FPO__ -D__SIM_OPENCV__ -D__SIM_FFT__ -D__SIM_FIR__ -D__SIM_DDS__ -D__DSP48E1__
CPPFLAGS += -g -O0
CPPFLAGS += -MMD -MP 
CXXFLAGS += -lm
CXXFLAGS += -std=c++14 -Wno-unused-result
LDLIBS += -lstdc++ -lm
LDLIBS += -L"${AUTOPILOT_ROOT}/lnx64/lib/csim" -lhlsmc++-GCC46 -Wl,-rpath,"${AUTOPILOT_ROOT}/lnx64/lib/csim" -Wl,-rpath,"${AUTOPILOT_ROOT}/lnx64/tools/fpo_v7_0"

CAT_TARGET = src/_single_file.cpp
SRCS = $(filter-out $(CAT_TARGET), $(wildcard **/*.cpp))
OBJS = $(SRCS:.cpp=.o)
DEPS = $(SRCS:.cpp=.d)

all: cat result

cat: $(CAT_TARGET)

$(CAT_TARGET): $(filter src/%, $(SRCS))
	: > $@
	$(foreach file, $^, echo '#line 1 "$(file)"' >> $@; cat "$(file)" >> $@;)

result: $(OBJS)
	$(CC) $(LDFLAGS) $^ $(LOADLIBES) $(LDLIBS) -o $@

.PHONY: clean

clean:
	$(RM) $(CAT_TARGET) $(OBJS) $(DEPS) result

-include $(DEPS)
