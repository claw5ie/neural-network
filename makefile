WFLAGS=-Wall -Wextra -pedantic

debug: ./main.cpp
	g++ $(WFLAGS) -g -std=c++11 $^

release: ./main.cpp
	g++ -O3 -D NDEBUG $^
