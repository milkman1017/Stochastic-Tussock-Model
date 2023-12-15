tussock_model: tussock_model.cpp tussock_model.h
	g++ -std=c++17 -Wall tussock_model.cpp tussock_model.h -fsanitize=thread -o tussock_model -lpthread
