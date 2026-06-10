CXX = g++
SRCDIR = src
BUILDDIR = build

# Auto-detect Eigen on common install paths. Override with:
#   make EIGEN_INCLUDE=-I/path/to/eigen3
EIGEN_INCLUDE ?=
ifeq ($(EIGEN_INCLUDE),)
  EIGEN_PREFIX := $(shell brew --prefix eigen 2>/dev/null)
  ifneq ($(EIGEN_PREFIX),)
    EIGEN_INCLUDE := -I$(EIGEN_PREFIX)/include/eigen3
  else ifneq ($(wildcard /usr/include/eigen3/Eigen/Dense),)
    EIGEN_INCLUDE := -I/usr/include/eigen3
  else ifneq ($(wildcard /opt/homebrew/include/eigen3/Eigen/Dense),)
    EIGEN_INCLUDE := -I/opt/homebrew/include/eigen3
  endif
endif

CXXFLAGS = -Isrc $(EIGEN_INCLUDE)

ifeq ($(EIGEN_INCLUDE),)
  $(warning Eigen not found. Install with: brew install eigen  OR  apt install libeigen3-dev)
  $(warning Or pass: make EIGEN_INCLUDE=-I/path/to/eigen3)
endif

.PHONY: all nn bb clean

all: nn bb

nn: $(BUILDDIR)/read_data.o $(BUILDDIR)/utils.o $(BUILDDIR)/neural_network.o $(BUILDDIR)/nn.o
	$(CXX) $(CXXFLAGS) $^ -o $(BUILDDIR)/nn

bb: $(BUILDDIR)/utils.o $(BUILDDIR)/bb.o
	$(CXX) $(CXXFLAGS) $^ -o $(BUILDDIR)/bb

$(BUILDDIR)/read_data.o: $(SRCDIR)/read_data.cpp $(SRCDIR)/read_data.hpp $(SRCDIR)/define.hpp
	@mkdir -p $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $(SRCDIR)/read_data.cpp -o $@

$(BUILDDIR)/utils.o: $(SRCDIR)/utils.cpp $(SRCDIR)/utils.hpp $(SRCDIR)/define.hpp
	@mkdir -p $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $(SRCDIR)/utils.cpp -o $@

$(BUILDDIR)/neural_network.o: $(SRCDIR)/neural_network.cpp $(SRCDIR)/neural_network.hpp $(SRCDIR)/utils.hpp $(SRCDIR)/define.hpp
	@mkdir -p $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $(SRCDIR)/neural_network.cpp -o $@

$(BUILDDIR)/bb.o: $(SRCDIR)/bb.cpp $(SRCDIR)/define.hpp $(SRCDIR)/utils.hpp
	@mkdir -p $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $(SRCDIR)/bb.cpp -o $@

$(BUILDDIR)/nn.o: $(SRCDIR)/nn.cpp $(SRCDIR)/define.hpp $(SRCDIR)/read_data.hpp $(SRCDIR)/utils.hpp $(SRCDIR)/neural_network.hpp
	@mkdir -p $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $(SRCDIR)/nn.cpp -o $@

clean:
	$(RM) -r $(BUILDDIR)
