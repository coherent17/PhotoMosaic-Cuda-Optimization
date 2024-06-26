# ompiler & Linker settings
CXX = nvcc
CXXFLAGS = -I ./inc -I ./third-party/CImg -I ./third-party/libjpeg -I ./Data-Loader -std=c++11 -Xptxas -O3
LINKER = -L/usr/X11R6/lib -lm -lX11 -L./third-party/libjpeg -ljpeg -lpng

# Valgrind for memory issue
CHECKCC = valgrind
CHECKFLAGS = --leak-check=full -s --show-leak-kinds=all --track-origins=yes 

# Source files and object files
SRCDIR = src
OBJDIR = obj
INCDIR = inc
SRCS = $(wildcard $(SRCDIR)/*.cpp)
OBJS = $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(SRCS))
DEPS = $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.d,$(SRCS))

# Control the build verbosity
ifeq ("$(VERBOSE)","1")
    Q :=
    VECHO = @true
else
    Q := @
    VECHO = @printf
endif

.PHONY: all install check clean

# Name of the executable
TARGET = PhotoMosaic

all: $(TARGET)

$(OBJDIR):
	@mkdir $(OBJDIR)

PhotoMosaic: main.cu $(OBJS) $(OBJDIR)/photo_mosaic_cuda.o
	$(VECHO) "	LD\t$@\n"
	$(Q)$(CXX) $(CXXFLAGS) $^ -o $@ $(LINKER)


# Include generated dependency files
-include $(DEPS)

# Compilation rule for object files with automatic dependency generation
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR) Makefile
	$(VECHO) "	NVCC\t$@\n"
	$(Q)$(CXX) $(CXXFLAGS) -MMD -c $< -o $@

# Compilation rule for object file with cuda
$(OBJDIR)/photo_mosaic_cuda.o: $(SRCDIR)/photo_mosaic_cuda.cu | $(OBJDIR) Makefile
	$(VECHO) "	NVCC\t$@\n"
	$(Q)$(CXX) $(CXXFLAGS) -MMD -c $< -o $@

install:
	$(VECHO) "Installing third party dependencies\n"
	$(Q)chmod +x scripts/clone_env.sh  
	$(Q)./scripts/clone_env.sh  > /dev/null 2>&1
	$(VECHO) "Finished installing third party dependencies!!\n"

check:
	$(CHECKCC) $(CHECKFLAGS) ./PhotoMosaic

clean:
	rm -rf $(OBJDIR) $(TARGET) *.jpg
