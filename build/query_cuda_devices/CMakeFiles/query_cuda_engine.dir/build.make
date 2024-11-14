# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /data/jmtang/miniforge3/envs/baichuan/lib/python3.10/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /data/jmtang/miniforge3/envs/baichuan/lib/python3.10/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/jmtang/codes/cuda_demos

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/jmtang/codes/cuda_demos/build

# Include any dependencies generated for this target.
include query_cuda_devices/CMakeFiles/query_cuda_engine.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include query_cuda_devices/CMakeFiles/query_cuda_engine.dir/compiler_depend.make

# Include the progress variables for this target.
include query_cuda_devices/CMakeFiles/query_cuda_engine.dir/progress.make

# Include the compile flags for this target's objects.
include query_cuda_devices/CMakeFiles/query_cuda_engine.dir/flags.make

query_cuda_devices/CMakeFiles/query_cuda_engine.dir/query_cuda_engine.cu.o: query_cuda_devices/CMakeFiles/query_cuda_engine.dir/flags.make
query_cuda_devices/CMakeFiles/query_cuda_engine.dir/query_cuda_engine.cu.o: /home/jmtang/codes/cuda_demos/query_cuda_devices/query_cuda_engine.cu
query_cuda_devices/CMakeFiles/query_cuda_engine.dir/query_cuda_engine.cu.o: query_cuda_devices/CMakeFiles/query_cuda_engine.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/jmtang/codes/cuda_demos/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CUDA object query_cuda_devices/CMakeFiles/query_cuda_engine.dir/query_cuda_engine.cu.o"
	cd /home/jmtang/codes/cuda_demos/build/query_cuda_devices && /usr/bin/nvcc -forward-unknown-to-host-compiler $(CUDA_DEFINES) $(CUDA_INCLUDES) $(CUDA_FLAGS) -MD -MT query_cuda_devices/CMakeFiles/query_cuda_engine.dir/query_cuda_engine.cu.o -MF CMakeFiles/query_cuda_engine.dir/query_cuda_engine.cu.o.d -x cu -c /home/jmtang/codes/cuda_demos/query_cuda_devices/query_cuda_engine.cu -o CMakeFiles/query_cuda_engine.dir/query_cuda_engine.cu.o

query_cuda_devices/CMakeFiles/query_cuda_engine.dir/query_cuda_engine.cu.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CUDA source to CMakeFiles/query_cuda_engine.dir/query_cuda_engine.cu.i"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_PREPROCESSED_SOURCE

query_cuda_devices/CMakeFiles/query_cuda_engine.dir/query_cuda_engine.cu.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CUDA source to assembly CMakeFiles/query_cuda_engine.dir/query_cuda_engine.cu.s"
	$(CMAKE_COMMAND) -E cmake_unimplemented_variable CMAKE_CUDA_CREATE_ASSEMBLY_SOURCE

# Object files for target query_cuda_engine
query_cuda_engine_OBJECTS = \
"CMakeFiles/query_cuda_engine.dir/query_cuda_engine.cu.o"

# External object files for target query_cuda_engine
query_cuda_engine_EXTERNAL_OBJECTS =

query_cuda_devices/query_cuda_engine: query_cuda_devices/CMakeFiles/query_cuda_engine.dir/query_cuda_engine.cu.o
query_cuda_devices/query_cuda_engine: query_cuda_devices/CMakeFiles/query_cuda_engine.dir/build.make
query_cuda_devices/query_cuda_engine: query_cuda_devices/CMakeFiles/query_cuda_engine.dir/linkLibs.rsp
query_cuda_devices/query_cuda_engine: query_cuda_devices/CMakeFiles/query_cuda_engine.dir/objects1.rsp
query_cuda_devices/query_cuda_engine: query_cuda_devices/CMakeFiles/query_cuda_engine.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/jmtang/codes/cuda_demos/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CUDA executable query_cuda_engine"
	cd /home/jmtang/codes/cuda_demos/build/query_cuda_devices && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/query_cuda_engine.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
query_cuda_devices/CMakeFiles/query_cuda_engine.dir/build: query_cuda_devices/query_cuda_engine
.PHONY : query_cuda_devices/CMakeFiles/query_cuda_engine.dir/build

query_cuda_devices/CMakeFiles/query_cuda_engine.dir/clean:
	cd /home/jmtang/codes/cuda_demos/build/query_cuda_devices && $(CMAKE_COMMAND) -P CMakeFiles/query_cuda_engine.dir/cmake_clean.cmake
.PHONY : query_cuda_devices/CMakeFiles/query_cuda_engine.dir/clean

query_cuda_devices/CMakeFiles/query_cuda_engine.dir/depend:
	cd /home/jmtang/codes/cuda_demos/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/jmtang/codes/cuda_demos /home/jmtang/codes/cuda_demos/query_cuda_devices /home/jmtang/codes/cuda_demos/build /home/jmtang/codes/cuda_demos/build/query_cuda_devices /home/jmtang/codes/cuda_demos/build/query_cuda_devices/CMakeFiles/query_cuda_engine.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : query_cuda_devices/CMakeFiles/query_cuda_engine.dir/depend

