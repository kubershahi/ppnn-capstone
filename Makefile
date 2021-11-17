
#  -g    adds debugging information to the executable file
#  -Wall turns on most, but not all, compiler warnings

CC = g++
# CFLAGS  = -g -Wall

bb: utils.o bb.o
	$(CC) $(CFLAGS) utils.o bb.o -o bb

nn: read_data.o utils.o neural_network.o nn.o
	$(CC) $(CFLAGS) read_data.o utils.o neural_network.o nn.o -o nn

read_data.o: read_data.cpp read_data.hpp
	$(CC) $(CFLAGS) -c read_data.cpp

utils.o: utils.cpp utils.hpp
	$(CC) $(CFLAGS) -c utils.cpp

neural_network.o: neural_network.cpp neural_network.hpp
	$(CC) $(CFLAGS) -c neural_network.cpp

bb.o: bb.cpp define.hpp utils.hpp 
	$(CC) $(CFLAGS) -c bb.cpp

nn.o: nn.cpp define.hpp read_data.hpp utils.hpp neural_network.hpp 
	$(CC) $(CFLAGS) -c nn.cpp


# To start over from scratch, type 'make clean'. This removes the executable file, 
# as well as old .o objectfiles and *~ backup files:
clean: 
	$(RM) nn bb file *.o *~