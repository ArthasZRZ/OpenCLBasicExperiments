AMDSDK=/opt/AMDAPP/

hello:
	gcc -I$(AMDSDK)/include -L$(AMDSDK)/lib/x86_64 -o hello.o hello.c -Wl,-rpath,$(AMDSDK)/lib/x86_64 -lOpenCL 
