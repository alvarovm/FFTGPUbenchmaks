
SOURCES := $(wildcard *.f)
OBJECTS := $(patsubst %.f,%.o,$(SOURCES))

lib = libfft.a

# compiler tools
fc = gfortran
ld = gfortran
ar = ar
#fflag = -o2 -cpp -ffast-math -march=native -funroll-loops -fno-protect-parens -flto -fcheck=all -fallow-argument-mismatch 
fflag = 
lflag =

# targets
.PHONY: all
#all: $(FOBJ)

all: $(lib)

%.o: %.f
	$(fc) $(fflag) -c $*.f


$(lib): $(OBJECTS)
	$(ar) rcs libfft.a $(OBJECTS)
#	@mkdir -p $(lib_dir)
#	$(ld) -shared -fpic $(objs) -o $(lib_dir)lib$(lib).dll.a
#	@mkdir -p $(lib_dir)include
#	@cp -f $(obj_dir)*.mod $(lib_dir)include
#
clean:
	rm -f *.o
	rm -f *.a
#
#test:
#	@mkdir -p $(example_dir)
#	cd $(example_dir); ../$(exe_dir)$(exe)
