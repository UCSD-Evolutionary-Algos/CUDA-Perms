CC=nvcc
RM=rm
CFLAGS=-Iinclude -Iinclude/external -Llib

all: perms

perms: src/perms.cu include/config.h
	$(CC) $(CFLAGS) src/perms.cu -o perms

debug: src/perms.cu include/config.h
	$(CC) $(CFLAGS) -g -Og -Wall -pedantic src/perms.cu -o perms

clean:
	$(RM) perms

