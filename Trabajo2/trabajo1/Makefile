########################################################################
##
## archivo 'make' para compilar, enlazar y ejecutar
## invocar con 'make'
## genera archivos ejecutable 'practicaX', usando todos los .cpp/.cc/.c presentes
##
########################################################################

.SUFFIXES:
.PHONY: start, exec, all, compile, clean, tar

INC					:= ./inc
BIN					:= ./bin
OBJ					:= ./obj
SRC					:= ./src


target_name         := $(BIN)/main
opt_dbg_flag        := -g
exit_first          := -Wfatal-errors
warn_all            := -Wall

units_ext           := $(wildcard $(SRC)/*.cpp $(SRC)/*.cc $(SRC)/*.c)
units_ext						:= $(filter-out $(wildcard $(SRC)/practica*.cpp), $(units_ext))
headers             := $(wildcard $(INC)/*.hpp $(INC)/*.hh $(INC)/*.h)
##units_ext           :=  main practica1 error-ogl
comp_version		:= -std=c++11

# gl_libs_base        := $(pkg-config --libs opencv)
# gl_lib_file         :=

# gl_lib_file_nv      := /usr/lib/nvidia-331/libGL.so
# ifeq ($(wildcard $(gl_lib_file_nv)),$(gl_lib_file_nv))
#    gl_lib_file      := $(gl_lib_file_nv)
# endif

# gl_libs             := $(gl_libs_base) # $(gl_lib_file)
other_ld_libs       := `pkg-config --libs opencv`
gtk_c_flags         :=
gtk_ld_libs         :=


units               := $(basename $(units_ext))
objs                := $(addprefix $(OBJ)/, $(addsuffix .o, $(notdir $(units))))
c_flags             := `pkg-config --cflags opencv` -I $(INC) -I $(SRC) $(src_dir) $(opt_dbg_flag) $(exit_first) $(warn_all) $(gtk_c_flags) $(comp_version)
ld_libs             := $(gtk_ld_libs) $(gl_libs) $(other_ld_libs)

INTERACTIVE := $(shell [ -t 0 ] && echo 1)

start:
	@make --no-print-directory exec

exec: $(target_name)
	@echo "ejecutando " $(target_name) " ...."
	./$(target_name)

all:
	echo $(objs)
	make clean
	make compile

compile: $(target_name)
	@echo "compilando fuentes: " $(units_ext)
	@make --no-print-directory $(target_name)

$(target_name) : $(objs) | $(BIN)
	@if [ -n "$(INTERACTIVE)" ]; then\
        echo `tput bold`---------------------------------------------------------------;\
	else\
		echo ---------------------------------------------------------------;\
  fi

	@echo "Enlazando      :" $(target_name)
	@echo "Unidades(ext)  :" $(units_ext)
	@echo "Objetos        :" $(objs)

	@if [ -n "$(INTERACTIVE)" ]; then\
		tput sgr0;\
	fi
	g++ -o $(target_name) $(objs) $(ld_libs)
	@echo ---------------------------------------------------------------


$(OBJ)/%.o: $(SRC)/%.cpp $(headers) | $(OBJ)

	@if [ -n "$(INTERACTIVE)" ]; then\
				echo `tput bold`---------------------------------------------------------------;\
	else\
		echo ---------------------------------------------------------------;\
	fi

	@echo Compilando: $(notdir $<)

	@if [ -n "$(INTERACTIVE)" ]; then\
		tput sgr0;\
	fi
	@g++ $(c_flags) -c $< -o $@

$(OBJ)/%.o: $(SRC)/%.cc $(headers) | $(OBJ)

	@if [ -n "$(INTERACTIVE)" ]; then\
				echo `tput bold`---------------------------------------------------------------;\
	else\
		echo ---------------------------------------------------------------;\
	fi
	@echo Compilando: $(notdir $<)

	@if [ -n "$(INTERACTIVE)" ]; then\
		tput sgr0;\
	fi
	@g++ $(c_flags) -c $< -o $@

$(OBJ)/%.o: $(SRC)/%.c $(headers) | $(OBJ)

	@if [ -n "$(INTERACTIVE)" ]; then\
				echo `tput bold`---------------------------------------------------------------;\
	else\
		echo ---------------------------------------------------------------;\
	fi
	@echo Compilando: $(notdir $<)

	@if [ -n "$(INTERACTIVE)" ]; then\
		tput sgr0;\
	fi
	@g++ $(c_flags) -c $< -o $@

# Creation of binary directory
$(BIN):
	mkdir -p $(BIN)

# Creation of objects directory
$(OBJ):
	mkdir -p $(OBJ)

clean:
	rm -f $(OBJ)/*.o $(target_name)

tar:
	tar czvf archivos_prac_1.tgz *.c* *.h* *.ply makefile
