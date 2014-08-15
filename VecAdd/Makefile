AMDSDK=/opt/AMDAPP

vec_add:
	g++ -I$(AMDSDK)/include -L$(AMDSDK)/lib/x86_64 -o vec_add.o vec_add_host.cpp -DARR_SIZE=$(SIZE) -lOpenCL
